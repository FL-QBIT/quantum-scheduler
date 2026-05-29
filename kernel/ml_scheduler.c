/*
 * ml_scheduler.c
 * ML Scheduler Daemon — Production Grade
 * 
 * Three-tier process classification:
 *   CRITICAL   → never touch (NetworkManager, Xorg, dbus...)
 *   USER       → prioritize within safe bounds (UID >= 1000)
 *   BACKGROUND → can only lower, never raise
 *
 * Safety features:
 *   → Per-tier priority bounds (no runaway priority)
 *   → 60-second decay (priorities reset toward 0)
 *   → Critical process protection (system stays stable)
 *   → Hybrid rule + ML decisions
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
#define MAX_PIDS            512
#define STEP_INTERVAL_MS    500
#define DECAY_INTERVAL_SECS  60
#define LOG_FILE            "/tmp/ml_scheduler.log"

/* ── Process tiers ────────────────────────────────────────── */
#define PROC_TIER_CRITICAL    0   /* never touch  */
#define PROC_TIER_USER        1   /* prioritize   */
#define PROC_TIER_BACKGROUND  2   /* can lower    */

/* ── Priority bounds per tier ─────────────────────────────── */
#define USER_NICE_FLOOR      -10  /* most priority user proc can get  */
#define USER_NICE_CEILING      5  /* least priority user proc can get */
#define BG_NICE_FLOOR          0  /* background never above neutral   */
#define BG_NICE_CEILING       19  /* background can go fully low      */

/* ── Rule thresholds ──────────────────────────────────────── */
#define THRESH_STARVING_WAIT  0.001f
#define THRESH_CPU_ACTIVE     0.001f
#define THRESH_CPU_BOUND      0.20f
#define THRESH_CPU_HOG        0.01f
#define THRESH_CPU_EXTREME    0.05f
#define THRESH_MEM_IDLE       0.05f

/* ── Decision source ──────────────────────────────────────── */
#define SOURCE_RULE  0
#define SOURCE_ML    1

/* ── Action names ─────────────────────────────────────────── */
static const char *ACTION_NAMES[] = {
    "LOWER_PRIO", "RAISE_PRIO", "KEEP", "BATCH", "NORMAL"
};

/* ── Tier names ───────────────────────────────────────────── */
static const char *TIER_NAMES[] = {
    "CRITICAL", "USER", "BACKGROUND"
};

/* ── Global state ─────────────────────────────────────────── */
static volatile int  running             = 1;
static FILE         *log_file            = NULL;
static long          total_steps         = 0;
static long          total_decisions[5]  = {0};
static long          rule_decisions      = 0;
static long          ml_decisions        = 0;
static long          skipped_critical    = 0;
static time_t        last_decay          = 0;

/* ── PID history table ────────────────────────────────────── */
struct tracked_pid {
    int               pid;
    struct pid_history hist;
};

static struct tracked_pid tracked[MAX_PIDS];
static int                n_tracked = 0;

/* ── Critical process list ────────────────────────────────── */
/*
 * These processes support the entire user session.
 * Touching them breaks networking, display, audio, input.
 * They are NEVER selected for scheduling decisions.
 */
static const char *CRITICAL_PROCS[] = {
    /* Networking */
    "NetworkManager", "wpa_supplicant", "dhclient",
    "dhcpcd", "systemd-network", "connmand",
    /* Display */
    "Xorg", "Xwayland", "gdm", "lightdm", "sddm",
    "kwin", "mutter", "xfwm4",
    /* Audio */
    "pulseaudio", "pipewire", "pipewire-pulse",
    "wireplumber", "jackd",
    /* Core system */
    "systemd", "dbus-daemon", "dbus-launch",
    "udevd", "systemd-udevd",
    /* Input */
    "bluetoothd", "input-remapper",
    /* Security */
    "sshd", "gpg-agent", "polkitd",
    NULL
};

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

/* ── Process tier classification ──────────────────────────── */
static int get_proc_tier(int pid, char *comm_out, int comm_len)
{
    char path[64];
    char line[256];
    char comm[64] = "";
    int  uid      = -1;

    snprintf(path, sizeof(path), "/proc/%d/status", pid);
    FILE *f = fopen(path, "r");
    if (!f) return PROC_TIER_BACKGROUND;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Uid:", 4) == 0)
            sscanf(line + 4, "%d", &uid);
        if (strncmp(line, "Name:", 5) == 0) {
            sscanf(line + 5, "%63s", comm);
            comm[63] = '\0';
        }
    }
    fclose(f);

    /* Return comm to caller if requested */
    if (comm_out && comm_len > 0) {
        strncpy(comm_out, comm, comm_len - 1);
        comm_out[comm_len - 1] = '\0';
    }

    /* Check critical list first — regardless of UID */
    for (int i = 0; CRITICAL_PROCS[i] != NULL; i++) {
        if (strstr(comm, CRITICAL_PROCS[i])) {
            return PROC_TIER_CRITICAL;
        }
    }

    /* UID-based classification */
    if (uid >= 1000)
        return PROC_TIER_USER;

    return PROC_TIER_BACKGROUND;
}

/* ── Priority bounds ──────────────────────────────────────── */
static void get_limits(int tier, int *floor, int *ceiling)
{
    switch (tier) {
    case PROC_TIER_USER:
        *floor   = USER_NICE_FLOOR;
        *ceiling = USER_NICE_CEILING;
        break;
    case PROC_TIER_BACKGROUND:
        *floor   = BG_NICE_FLOOR;
        *ceiling = BG_NICE_CEILING;
        break;
    default:
        /* Critical — should never reach here */
        *floor   = -20;
        *ceiling =  19;
        break;
    }
}

/* ── Weighted PID selection ───────────────────────────────── */
static int pick_weighted_pid(void)
{
    int user_pids[128];
    int bg_pids[256];
    int n_user = 0;
    int n_bg   = 0;

    DIR           *d = opendir("/proc");
    struct dirent *e;
    if (!d) return -1;

    while ((e = readdir(d)) != NULL) {
        int pid = atoi(e->d_name);
        if (pid <= 2) continue;

        char check[32];
        snprintf(check, sizeof(check), "/proc/%d", pid);
        if (access(check, F_OK) != 0) continue;

        int tier = get_proc_tier(pid, NULL, 0);

        /* Never select critical processes */
        if (tier == PROC_TIER_CRITICAL)
            continue;

        if (tier == PROC_TIER_USER) {
            if (n_user < 128)
                user_pids[n_user++] = pid;
        } else {
            if (n_bg < 256)
                bg_pids[n_bg++] = pid;
        }
    }
    closedir(d);

    /* 80% user, 20% background */
    if (n_user > 0 && (rand() % 10) < 8)
        return user_pids[rand() % n_user];
    else if (n_bg > 0)
        return bg_pids[rand() % n_bg];

    return -1;
}

/* ── History lookup ───────────────────────────────────────── */
static struct pid_history *get_history(int pid)
{
    for (int i = 0; i < n_tracked; i++)
        if (tracked[i].pid == pid)
            return &tracked[i].hist;

    if (n_tracked < MAX_PIDS) {
        memset(&tracked[n_tracked], 0,
               sizeof(struct tracked_pid));
        tracked[n_tracked].pid = pid;
        return &tracked[n_tracked++].hist;
    }

    /* Evict oldest */
    memmove(&tracked[0], &tracked[1],
            (MAX_PIDS - 1) * sizeof(struct tracked_pid));
    n_tracked = MAX_PIDS - 1;
    memset(&tracked[n_tracked], 0,
           sizeof(struct tracked_pid));
    tracked[n_tracked].pid = pid;
    return &tracked[n_tracked++].hist;
}

/* ── Rule engine ──────────────────────────────────────────── */
static int apply_rules(const float features[ML_INPUT_DIM],
                       int tier)
{
    float cpu   = features[FEAT_CPU_USAGE];
    float wait  = features[FEAT_WAIT_TIME];
    float mem   = features[FEAT_MEMORY];
    float state = features[FEAT_STATE_PATTERN];

    /* Background processes can only be lowered or kept */
    if (tier == PROC_TIER_BACKGROUND) {
        if (cpu > THRESH_CPU_EXTREME)
            return ACTION_LOWER_PRIO;
        return ACTION_KEEP;
    }

    /* User process rules */

    /* Rule 1: Starving AND doing real work → raise */
    if (wait > THRESH_STARVING_WAIT &&
        cpu  > THRESH_CPU_ACTIVE)
        return ACTION_RAISE_PRIO;

    /* Rule 2: CPU-bound hog → batch */
    if (cpu   > THRESH_CPU_HOG &&
        state > THRESH_CPU_BOUND)
        return ACTION_BATCH;

    /* Rule 3: Extreme CPU → lower */
    if (cpu > THRESH_CPU_EXTREME)
        return ACTION_LOWER_PRIO;

    /* Rule 4: High memory, idle CPU → keep */
    if (mem > THRESH_MEM_IDLE &&
        cpu < THRESH_CPU_ACTIVE)
        return ACTION_KEEP;

    /* No rule matched → defer to ML */
    return -1;
}

/* ── Action application with bounds ──────────────────────── */
static int apply_action(int pid, int action, int tier)
{
    int nice_floor, nice_ceiling;
    get_limits(tier, &nice_floor, &nice_ceiling);

    errno = 0;
    int current_nice = getpriority(PRIO_PROCESS, pid);
    if (errno) return -1;

    int new_nice = current_nice;

    switch (action) {
    case ACTION_RAISE_PRIO:
        new_nice = current_nice - 5;
        break;
    case ACTION_LOWER_PRIO:
        new_nice = current_nice + 5;
        break;
    case ACTION_KEEP:
        return 0;
    case ACTION_BATCH:
        new_nice = current_nice + 3;
        break;
    case ACTION_NORMAL:
        new_nice = 0;
        break;
    default:
        return -1;
    }

    /* Clamp to tier bounds — prevents runaway priority */
    if (new_nice < nice_floor)   new_nice = nice_floor;
    if (new_nice > nice_ceiling) new_nice = nice_ceiling;

    /* Skip if no actual change */
    if (new_nice == current_nice)
        return 0;

    return setpriority(PRIO_PROCESS, pid, new_nice);
}

/* ── Priority decay ───────────────────────────────────────── */
/*
 * Every DECAY_INTERVAL_SECS seconds, nudge all
 * user processes one step back toward nice=0.
 *
 * Prevents permanent priority inversion.
 * Ensures scheduler re-evaluates continuously.
 */
static void decay_priorities(void)
{
    DIR           *d = opendir("/proc");
    struct dirent *e;
    int            decayed = 0;

    if (!d) return;

    while ((e = readdir(d)) != NULL) {
        int pid = atoi(e->d_name);
        if (pid <= 2) continue;

        int tier = get_proc_tier(pid, NULL, 0);
        if (tier != PROC_TIER_USER) continue;

        errno = 0;
        int nice = getpriority(PRIO_PROCESS, pid);
        if (errno) continue;

        /* Only decay if away from 0 */
        if (nice < 0) {
            setpriority(PRIO_PROCESS, pid, nice + 1);
            decayed++;
        } else if (nice > 0) {
            setpriority(PRIO_PROCESS, pid, nice - 1);
            decayed++;
        }
    }
    closedir(d);

    if (decayed > 0) {
        printf("\n  [decay] nudged %d user processes "
               "toward nice=0\n\n", decayed);
        log_msg("decay: nudged %d processes", decayed);
    }
}

/* ── Stats printer ────────────────────────────────────────── */
static void print_stats(void)
{
    printf("\n┌──────────────────────────────────────────────┐\n");
    printf("│  ML Scheduler Stats                          │\n");
    printf("├──────────────────────────────────────────────┤\n");
    printf("│  Total steps:       %8ld                 │\n",
           total_steps);
    printf("│  Tracked PIDs:      %8d                 │\n",
           n_tracked);
    printf("│  Skipped critical:  %8ld                 │\n",
           skipped_critical);
    printf("├──────────────────────────────────────────────┤\n");
    printf("│  Decision source:                            │\n");
    printf("│    Rule:  %6ld  (%4.1f%%)                   │\n",
           rule_decisions,
           total_steps > 0 ?
           (float)rule_decisions / total_steps * 100 : 0);
    printf("│    ML:    %6ld  (%4.1f%%)                   │\n",
           ml_decisions,
           total_steps > 0 ?
           (float)ml_decisions / total_steps * 100 : 0);
    printf("├──────────────────────────────────────────────┤\n");
    printf("│  Action distribution:                        │\n");
    for (int i = 0; i < 5; i++) {
        float pct = total_steps > 0 ?
            (float)total_decisions[i] / total_steps * 100 : 0;
        printf("│  %11s: %6ld  (%5.1f%%)              │\n",
               ACTION_NAMES[i],
               total_decisions[i], pct);
    }
    printf("└──────────────────────────────────────────────┘\n\n");
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
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║     ML Process Scheduler Daemon              ║\n");
    printf("║     Hybrid Rule-Based + Neural Net           ║\n");
    printf("║                                              ║\n");
    printf("║     Safety:  3-tier classification           ║\n");
    printf("║              per-tier priority bounds        ║\n");
    printf("║              60s decay toward neutral        ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    /* Verify inference engine */
    printf("Verifying inference engine...\n");
    if (!ml_verify()) {
        fprintf(stderr,
                "FATAL: inference verification failed\n");
        return 1;
    }

    /* Show tier config */
    printf("\nTier configuration:\n");
    printf("  CRITICAL   → never touched (system stability)\n");
    printf("  USER       → nice %d to %d (user experience)\n",
           USER_NICE_FLOOR, USER_NICE_CEILING);
    printf("  BACKGROUND → nice %d to %d (can be lowered)\n",
           BG_NICE_FLOOR, BG_NICE_CEILING);
    printf("  Decay:     → every %ds, nudge toward nice=0\n\n",
           DECAY_INTERVAL_SECS);

    printf("Log: %s\n", LOG_FILE);
    printf("Starting... (Ctrl+C to stop)\n\n");

    /* Header */
    printf("%-8s %-18s %-12s %-5s %-10s %-6s %-8s\n",
           "PID", "COMM", "ACTION",
           "SRC", "TIER", "NICE", "VALUE");
    printf("%-8s %-18s %-12s %-5s %-10s %-6s %-8s\n",
           "───────", "─────────────────",
           "───────────", "────",
           "─────────", "─────", "───────");

    log_msg("ML Scheduler started "
            "(3-tier, hybrid, bounded, decay)");

    last_decay = time(NULL);

    struct timespec interval = {
        .tv_sec  = 0,
        .tv_nsec = STEP_INTERVAL_MS * 1000000L
    };

    while (running) {

        /* ── Decay check ─────────────────────────────────── */
        time_t now = time(NULL);
        if (now - last_decay >= DECAY_INTERVAL_SECS) {
            decay_priorities();
            last_decay = now;
        }

        /* ── PID selection ───────────────────────────────── */
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

        /* Get tier and comm together */
        char comm[64] = "unknown";
        int  tier     = get_proc_tier(pid, comm, sizeof(comm));

        /* Skip critical — should not reach here from
         * pick_weighted_pid but double-check for safety */
        if (tier == PROC_TIER_CRITICAL) {
            skipped_critical++;
            nanosleep(&interval, NULL);
            continue;
        }

        /* ── Feature extraction ──────────────────────────── */
        struct pid_history *hist = get_history(pid);
        float features[ML_INPUT_DIM];
        ml_extract_features(pid, hist, features);

        /* ── Decision: rules first, ML fallback ─────────── */
        int final_action;
        int source;

        int rule_action = apply_rules(features, tier);

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

        /* ── Read current nice ───────────────────────────── */
        errno = 0;
        int current_nice = getpriority(PRIO_PROCESS, pid);
        if (errno) {
            nanosleep(&interval, NULL);
            continue;
        }

        /* ── Apply with bounds ───────────────────────────── */
        apply_action(pid, final_action, tier);

        /* ── Read new nice ───────────────────────────────── */
        errno = 0;
        int new_nice = getpriority(PRIO_PROCESS, pid);
        if (errno) new_nice = current_nice;

        /* ── Value estimate ──────────────────────────────── */
        struct ml_result r = ml_infer(features);

        /* ── Color coding ────────────────────────────────── */
        const char *action_color;
        switch (final_action) {
        case ACTION_RAISE_PRIO: action_color = "\033[92m"; break;
        case ACTION_LOWER_PRIO: action_color = "\033[91m"; break;
        case ACTION_BATCH:      action_color = "\033[93m"; break;
        case ACTION_NORMAL:     action_color = "\033[96m"; break;
        default:                action_color = "\033[0m";  break;
        }

        const char *src_str = (source == SOURCE_RULE) ?
                              "\033[93mRULE\033[0m" :
                              "\033[96mML  \033[0m";

        const char *tier_color;
        switch (tier) {
        case PROC_TIER_USER:       tier_color = "\033[92m"; break;
        case PROC_TIER_BACKGROUND: tier_color = "\033[94m"; break;
        default:                   tier_color = "\033[0m";  break;
        }

        /* ── Print ───────────────────────────────────────── */
        printf("%-8d %-18s %s%-12s\033[0m %-5s "
               "%s%-10s\033[0m %-6d v=%.3f\n",
               pid, comm,
               action_color, ACTION_NAMES[final_action],
               src_str,
               tier_color, TIER_NAMES[tier],
               new_nice,
               r.value);

        /* Show features for non-KEEP decisions */
        if (final_action != ACTION_KEEP)
            print_features_brief(features);

        fflush(stdout);

        /* ── Log ─────────────────────────────────────────── */
        log_msg("pid=%d comm=%s tier=%s action=%s "
                "src=%s nice=%d->%d "
                "cpu=%.3f wait=%.3f",
                pid, comm,
                TIER_NAMES[tier],
                ACTION_NAMES[final_action],
                source == SOURCE_RULE ? "RULE" : "ML",
                current_nice, new_nice,
                features[FEAT_CPU_USAGE],
                features[FEAT_WAIT_TIME]);

        total_decisions[final_action]++;
        total_steps++;

        if (total_steps % 50 == 0)
            print_stats();

        nanosleep(&interval, NULL);
    }

    /* ── Shutdown ────────────────────────────────────────── */
    printf("\nShutting down...\n");
    print_stats();
    log_msg("Stopped. Total steps: %ld  "
            "Rule: %ld  ML: %ld",
            total_steps, rule_decisions, ml_decisions);

    if (log_file != stderr)
        fclose(log_file);

    return 0;
}
