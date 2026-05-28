/*
 * ml_scheduler.c
 * ML Scheduler Daemon — Hybrid Rule-Based + ML approach.
 * Uses UID-based user process detection — no hardcoding.
 * Production approach — same as Google/Meta schedulers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include "ml_inference.h"
#include "ml_features.h"

/* ── Config ───────────────────────────────────────────────── */
#define MAX_PIDS          512
#define STEP_INTERVAL_MS  500
#define LOG_FILE          "/tmp/ml_scheduler.log"

/* ── Rule thresholds ──────────────────────────────────────── */
#define THRESH_STARVING_WAIT   0.001f
#define THRESH_CPU_ACTIVE      0.001f
#define THRESH_CPU_BOUND       0.30f
#define THRESH_CPU_HOG         0.02f
#define THRESH_CPU_EXTREME     0.10f
#define THRESH_MEM_IDLE        0.05f

/* ── Decision source ──────────────────────────────────────── */
#define SOURCE_RULE  0
#define SOURCE_ML    1

/* ── Action names ─────────────────────────────────────────── */
static const char *ACTION_NAMES[] = {
    "LOWER_PRIO", "RAISE_PRIO", "KEEP", "BATCH", "NORMAL"
};

/* ── Global state ─────────────────────────────────────────── */
static volatile int  running          = 1;
static FILE         *log_file         = NULL;
static long          total_steps      = 0;
static long          total_decisions[5]  = {0};
static long          rule_decisions      = 0;
static long          ml_decisions        = 0;

/* ── PID history table ────────────────────────────────────── */
struct tracked_pid {
    int               pid;
    struct pid_history hist;
};

static struct tracked_pid tracked[MAX_PIDS];
static int                n_tracked = 0;

/* ── Logging ──────────────────────────────────────────────── */
static void log_msg(const char *fmt, ...)
{
    if (!log_file) return;
    time_t     t  = time(NULL);
    struct tm *tm = localtime(&t);
    fprintf(log_file, "[%02d:%02d:%02d] ",
            tm->tm_hour, tm->tm_min, tm->tm_sec);
    va_list args;
    va_start(args, fmt);
    vfprintf(log_file, fmt, args);
    va_end(args);
    fprintf(log_file, "\n");
    fflush(log_file);
}

/* ── Signal handler ───────────────────────────────────────── */
static void shutdown_handler(int sig)
{
    (void)sig;
    running = 0;
}

/* ── UID-based user process detection ────────────────────── */
/*
 * Returns 1 if process is owned by a real user (UID >= 1000).
 * No hardcoding — works for ANY app the user runs.
 *
 * UID 0-999   → kernel threads, system daemons → low priority
 * UID 1000+   → real user processes            → high priority
 */
static int is_user_proc(int pid)
{
    char path[64];
    char line[256];
    int  uid = -1;

    snprintf(path, sizeof(path), "/proc/%d/status", pid);
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Uid:", 4) == 0) {
            sscanf(line + 4, "%d", &uid);
            break;
        }
    }
    fclose(f);

    return (uid >= 1000);
}

/* ── Weighted PID selection ───────────────────────────────── */
/*
 * 80% chance: pick a user process (UID >= 1000)
 * 20% chance: pick a system process
 * No names hardcoded — purely UID based.
 */
static int pick_weighted_pid(void)
{
    int user_pids[128];
    int system_pids[256];
    int n_user   = 0;
    int n_system = 0;

    DIR           *d = opendir("/proc");
    struct dirent *e;
    if (!d) return -1;

    while ((e = readdir(d)) != NULL) {
        int pid = atoi(e->d_name);
        if (pid <= 2) continue;

        /* Check process still alive */
        char check[32];
        snprintf(check, sizeof(check), "/proc/%d", pid);
        if (access(check, F_OK) != 0) continue;

        if (is_user_proc(pid)) {
            if (n_user < 128)
                user_pids[n_user++] = pid;
        } else {
            if (n_system < 256)
                system_pids[n_system++] = pid;
        }
    }
    closedir(d);

    /* Weighted selection */
    if (n_user > 0 && (rand() % 10) < 8)
        return user_pids[rand() % n_user];
    else if (n_system > 0)
        return system_pids[rand() % n_system];

    return -1;
}

/* ── History lookup ───────────────────────────────────────── */
/*
 * Find or create history entry for a PID.
 * History tracks recent scheduler events for state_pattern
 * and cpu_affinity features.
 */
static struct pid_history *get_history(int pid)
{
    /* Search existing */
    for (int i = 0; i < n_tracked; i++)
        if (tracked[i].pid == pid)
            return &tracked[i].hist;

    /* Add new entry */
    if (n_tracked < MAX_PIDS) {
        memset(&tracked[n_tracked], 0,
               sizeof(struct tracked_pid));
        tracked[n_tracked].pid = pid;
        return &tracked[n_tracked++].hist;
    }

    /* Table full — evict oldest entry */
    memmove(&tracked[0], &tracked[1],
            (MAX_PIDS - 1) * sizeof(struct tracked_pid));
    n_tracked = MAX_PIDS - 1;
    memset(&tracked[n_tracked], 0,
           sizeof(struct tracked_pid));
    tracked[n_tracked].pid = pid;
    return &tracked[n_tracked++].hist;
}

/* ── Rule engine ──────────────────────────────────────────── */
/*
 * Applies scheduling rules based on feature values.
 * Returns action (0-4) or -1 if no rule matched.
 *
 * Rule priority order:
 *   1. Starvation + active CPU use  → RAISE_PRIO
 *   2. CPU-bound hog                → BATCH
 *   3. Extreme CPU usage            → LOWER_PRIO
 *   4. Idle memory hog              → KEEP (explicit)
 *   5. No match                     → defer to ML (-1)
 */
static int apply_rules(const float features[ML_INPUT_DIM])
{
    float cpu   = features[FEAT_CPU_USAGE];
    float wait  = features[FEAT_WAIT_TIME];
    float mem   = features[FEAT_MEMORY];
    float state = features[FEAT_STATE_PATTERN];

    /* Rule 1: Being starved AND doing real work */
    if (wait > THRESH_STARVING_WAIT &&
        cpu  > THRESH_CPU_ACTIVE)
        return ACTION_RAISE_PRIO;

    /* Rule 2: CPU-bound hog */
    if (cpu   > THRESH_CPU_HOG &&
        state > THRESH_CPU_BOUND)
        return ACTION_BATCH;

    /* Rule 3: Extreme CPU usage */
    if (cpu > THRESH_CPU_EXTREME)
        return ACTION_LOWER_PRIO;

    /* Rule 4: High memory, idle CPU */
    if (mem > THRESH_MEM_IDLE &&
        cpu < THRESH_CPU_ACTIVE)
        return ACTION_KEEP;

    /* No rule matched */
    return -1;
}

/* ── Action application ───────────────────────────────────── */
static int apply_action(int pid, int action)
{
    int current_nice;
    int new_nice;

    errno        = 0;
    current_nice = getpriority(PRIO_PROCESS, pid);
    if (errno) return -1;

    switch (action) {
    case ACTION_LOWER_PRIO:
        new_nice = current_nice + 5;
        if (new_nice > 19) new_nice = 19;
        break;
    case ACTION_RAISE_PRIO:
        new_nice = current_nice - 5;
        if (new_nice < -5) new_nice = -5;
        break;
    case ACTION_KEEP:
        return 0;
    case ACTION_BATCH:
        new_nice = current_nice + 3;
        if (new_nice > 19) new_nice = 19;
        break;
    case ACTION_NORMAL:
        new_nice = 0;
        break;
    default:
        return -1;
    }

    return setpriority(PRIO_PROCESS, pid, new_nice);
}

/* ── Stats printer ────────────────────────────────────────── */
static void print_stats(void)
{
    printf("\n┌─────────────────────────────────────────┐\n");
    printf("│  ML Scheduler Stats                     │\n");
    printf("├─────────────────────────────────────────┤\n");
    printf("│  Total steps:    %8ld               │\n",
           total_steps);
    printf("│  Tracked PIDs:   %8d               │\n",
           n_tracked);
    printf("│  Rule decisions: %8ld  (%4.1f%%)       │\n",
           rule_decisions,
           total_steps > 0 ?
           (float)rule_decisions / total_steps * 100 : 0);
    printf("│  ML decisions:   %8ld  (%4.1f%%)       │\n",
           ml_decisions,
           total_steps > 0 ?
           (float)ml_decisions / total_steps * 100 : 0);
    printf("├─────────────────────────────────────────┤\n");
    printf("│  Action distribution:                   │\n");
    for (int i = 0; i < 5; i++) {
        float pct = total_steps > 0 ?
            (float)total_decisions[i] / total_steps * 100 : 0;
        printf("│  %11s: %6ld  (%5.1f%%)          │\n",
               ACTION_NAMES[i],
               total_decisions[i], pct);
    }
    printf("└─────────────────────────────────────────┘\n\n");
}

/* ── Feature brief printer ────────────────────────────────── */
static void print_features_brief(const float f[ML_INPUT_DIM])
{
    printf("  [cpu=%.3f io=%.3f wait=%.3f "
           "mem=%.3f state=%.2f]\n",
           f[FEAT_CPU_USAGE],
           f[FEAT_IO_RATIO],
           f[FEAT_WAIT_TIME],
           f[FEAT_MEMORY],
           f[FEAT_STATE_PATTERN]);
}

/* ── Main ─────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    srand((unsigned)time(NULL));

    signal(SIGINT,  shutdown_handler);
    signal(SIGTERM, shutdown_handler);

    log_file = fopen(LOG_FILE, "a");
    if (!log_file)
        log_file = stderr;

    /* Banner */
    printf("╔══════════════════════════════════════════╗\n");
    printf("║     ML Process Scheduler Daemon          ║\n");
    printf("║     Hybrid Rule-Based + Neural Net       ║\n");
    printf("║     UID-based user process detection     ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    /* Verify inference engine */
    printf("Verifying inference engine...\n");
    if (!ml_verify()) {
        fprintf(stderr,
                "FATAL: inference verification failed\n");
        return 1;
    }

    printf("\nLog: %s\n", LOG_FILE);
    printf("Starting... (Ctrl+C to stop)\n\n");

    /* Header */
    printf("%-8s %-18s %-12s %-6s %-6s %-8s\n",
           "PID", "COMM", "ACTION", "SRC", "NICE", "VALUE");
    printf("%-8s %-18s %-12s %-6s %-6s %-8s\n",
           "───────", "─────────────────",
           "───────────", "─────",
           "─────", "───────");

    log_msg("ML Scheduler started (UID-based, hybrid mode)");

    struct timespec interval = {
        .tv_sec  = 0,
        .tv_nsec = STEP_INTERVAL_MS * 1000000L
    };

    while (running) {

        /* Pick PID — weighted toward user processes */
        int pid = pick_weighted_pid();
        if (pid <= 0) {
            nanosleep(&interval, NULL);
            continue;
        }

        /* Check alive */
        char check[32];
        snprintf(check, sizeof(check), "/proc/%d", pid);
        if (access(check, F_OK) != 0) {
            nanosleep(&interval, NULL);
            continue;
        }

        /* Read process name */
        char comm[32] = "unknown";
        char comm_path[48];
        snprintf(comm_path, sizeof(comm_path),
                 "/proc/%d/comm", pid);
        FILE *cf = fopen(comm_path, "r");
        if (cf) {
            if (fgets(comm, sizeof(comm), cf))
                comm[strcspn(comm, "\n")] = 0;
            fclose(cf);
        }

        /* Get or create history for this PID */
        struct pid_history *hist = get_history(pid);

        /* Extract features */
        float features[ML_INPUT_DIM];
        ml_extract_features(pid, hist, features);

        /* ── Decision: rules first, ML fallback ─────────── */
        int final_action;
        int source;

        int rule_action = apply_rules(features);

        if (rule_action >= 0) {
            final_action = rule_action;
            source       = SOURCE_RULE;
            rule_decisions++;
        } else {
            struct ml_result r = ml_infer(features);
            final_action       = r.action;
            source             = SOURCE_ML;
            ml_decisions++;
        }

        /* Read current nice */
        errno = 0;
        int current_nice = getpriority(PRIO_PROCESS, pid);
        if (errno) {
            nanosleep(&interval, NULL);
            continue;
        }

        /* Apply decision */
        apply_action(pid, final_action);

        /* Read new nice */
        errno = 0;
        int new_nice = getpriority(PRIO_PROCESS, pid);
        if (errno) new_nice = current_nice;

        /* Run inference for value display */
        struct ml_result r = ml_infer(features);

        /* Color coding */
        const char *action_color;
        switch (final_action) {
        case ACTION_RAISE_PRIO: action_color = "\033[92m"; break;
        case ACTION_LOWER_PRIO: action_color = "\033[91m"; break;
        case ACTION_BATCH:      action_color = "\033[93m"; break;
        case ACTION_NORMAL:     action_color = "\033[96m"; break;
        default:                action_color = "\033[0m";  break;
        }

        const char *src_color = (source == SOURCE_RULE) ?
                                "\033[93mRULE\033[0m" :
                                "\033[96mML  \033[0m";

        /* Print decision */
        printf("%-8d %-18s %s%-12s\033[0m %-6s "
               "%-6d v=%.3f\n",
               pid, comm,
               action_color, ACTION_NAMES[final_action],
               src_color,
               new_nice,
               r.value);

        /* Show features for non-KEEP decisions */
        if (final_action != ACTION_KEEP)
            print_features_brief(features);

        fflush(stdout);

        log_msg("pid=%d comm=%s action=%s src=%s "
                "nice=%d cpu=%.3f wait=%.3f",
                pid, comm,
                ACTION_NAMES[final_action],
                source == SOURCE_RULE ? "RULE" : "ML",
                new_nice,
                features[FEAT_CPU_USAGE],
                features[FEAT_WAIT_TIME]);

        total_decisions[final_action]++;
        total_steps++;

        if (total_steps % 50 == 0)
            print_stats();

        nanosleep(&interval, NULL);
    }

    /* Shutdown */
    printf("\nShutting down...\n");
    print_stats();
    log_msg("Stopped. Total steps: %ld", total_steps);

    if (log_file != stderr)
        fclose(log_file);

    return 0;
}
