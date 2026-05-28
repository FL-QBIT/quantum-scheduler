/*
 * ml_features.c
 * C implementation of the feature extractor.
 * Reads /proc/<pid>/stat, schedstat, io, status.
 * Matches Python FeatureExtractor output exactly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ml_features.h"
#include <unistd.h>    /* add this line — needed for getpid() */

/* Default feature vector for cold start */
static const float DEFAULT_FEATURES[ML_INPUT_DIM] = {
    0.5f,   /* cpu_usage     */
    0.1f,   /* io_ratio      */
    0.3f,   /* wait_time     */
    0.5f,   /* burst_len     */
    0.1f,   /* timeslices    */
    0.2f,   /* memory        */
    0.86f,  /* priority      */
    0.1f,   /* threads       */
    0.5f,   /* state_pattern */
    1.0f,   /* cpu_affinity  */
};

/* ── /proc readers ────────────────────────────────────────── */

typedef struct {
    char   state;
    long   utime;
    long   stime;
    int    priority;
    int    nice;
    int    threads;
} proc_stat_t;

typedef struct {
    long long cpu_time_ns;
    long long wait_time_ns;
    long long timeslices;
} proc_schedstat_t;

typedef struct {
    long long read_bytes;
    long long write_bytes;
} proc_io_t;

typedef struct {
    long vm_rss_kb;
} proc_status_t;

static int read_stat(int pid, proc_stat_t *out)
{
    char path[64];
    char buf[512];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    /* stat format: pid (comm) state ... */
    /* skip past the comm field which may contain spaces */
    char *line = fgets(buf, sizeof(buf), f);
    fclose(f);
    if (!line) return -1;

    /* Find closing ) of comm field */
    char *p = strrchr(buf, ')');
    if (!p) return -1;
    p += 2;  /* skip ') ' */

    long utime, stime;
    int  priority, nice, threads;
    char state;

    /* Fields after comm: state ppid pgrp ... utime stime ... */
    sscanf(p,
           "%c "           /* state       field 3  */
           "%*d %*d %*d "  /* ppid pgrp session    */
           "%*d %*d %*d "  /* tty_nr tpgid flags   */
           "%*u %*u %*u %*u " /* minflt cminflt majflt cmajflt */
           "%ld %ld "      /* utime stime fields 14-15 */
           "%*d %*d "      /* cutime cstime        */
           "%d %d "        /* priority nice        */
           "%d",           /* num_threads          */
           &state,
           &utime, &stime,
           &priority, &nice,
           &threads);

    out->state    = state;
    out->utime    = utime;
    out->stime    = stime;
    out->priority = priority;
    out->nice     = nice;
    out->threads  = threads;
    return 0;
}

static int read_schedstat(int pid, proc_schedstat_t *out)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/schedstat", pid);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    int ok = fscanf(f, "%lld %lld %lld",
                    &out->cpu_time_ns,
                    &out->wait_time_ns,
                    &out->timeslices);
    fclose(f);
    return (ok == 3) ? 0 : -1;
}

static int read_io(int pid, proc_io_t *out)
{
    char path[64];
    char line[128];
    snprintf(path, sizeof(path), "/proc/%d/io", pid);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    out->read_bytes  = 0;
    out->write_bytes = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "read_bytes:", 11) == 0)
            sscanf(line + 11, "%lld", &out->read_bytes);
        else if (strncmp(line, "write_bytes:", 12) == 0)
            sscanf(line + 12, "%lld", &out->write_bytes);
    }
    fclose(f);
    return 0;
}

static int read_status(int pid, proc_status_t *out)
{
    char path[64];
    char line[128];
    snprintf(path, sizeof(path), "/proc/%d/status", pid);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    out->vm_rss_kb = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &out->vm_rss_kb);
            break;
        }
    }
    fclose(f);
    return 0;
}

static long get_total_ram_kb(void)
{
    char line[128];
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 8 * 1024 * 1024;  /* fallback 8GB */

    long total = 8 * 1024 * 1024;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%ld", &total);
            break;
        }
    }
    fclose(f);
    return total;
}

/* ── History helpers ──────────────────────────────────────── */

void ml_history_update(struct pid_history *hist,
                       int pid, int state, int cpu)
{
    if (hist->pid != pid) {
        /* Reset history for new PID */
        memset(hist, 0, sizeof(*hist));
        hist->pid = pid;
    }
    hist->states[hist->head] = state;
    hist->cpus[hist->head]   = cpu;
    hist->head  = (hist->head + 1) % HISTORY_LEN;
    if (hist->count < HISTORY_LEN)
        hist->count++;
}

static float compute_state_pattern(struct pid_history *hist)
{
    if (hist->count == 0)
        return 0.5f;

    int cpu_bound = 0;
    int n         = hist->count;

    for (int i = 0; i < n; i++)
        if (hist->states[i] == 0)  /* 0 = R = running/preempted */
            cpu_bound++;

    return (float)cpu_bound / (float)n;
}

static float compute_affinity(struct pid_history *hist)
{
    if (hist->count == 0)
        return 1.0f;

    /* Count unique CPUs used */
    int seen[256] = {0};
    int unique    = 0;

    for (int i = 0; i < hist->count; i++) {
        int cpu = hist->cpus[i];
        if (cpu >= 0 && cpu < 256 && !seen[cpu]) {
            seen[cpu] = 1;
            unique++;
        }
    }

    return 1.0f / (float)(unique > 0 ? unique : 1);
}

/* ── Main feature extraction ──────────────────────────────── */

int ml_extract_features(int pid,
                        struct pid_history *hist,
                        float features[ML_INPUT_DIM])
{
    static long total_ram_kb = 0;
    if (total_ram_kb == 0)
        total_ram_kb = get_total_ram_kb();

    proc_stat_t      stat      = {0};
    proc_schedstat_t schedstat = {0};
    proc_io_t        io        = {0};
    proc_status_t    status    = {0};

    int has_stat      = (read_stat(pid, &stat)           == 0);
    int has_schedstat = (read_schedstat(pid, &schedstat) == 0);
    int has_io        = (read_io(pid, &io)               == 0);
    int has_status    = (read_status(pid, &status)       == 0);

    /* Cold start — process just appeared */
    if (!has_stat && !has_schedstat) {
        memcpy(features, DEFAULT_FEATURES,
               ML_INPUT_DIM * sizeof(float));
        return 0;
    }

    /* Feature 0: CPU usage ratio */
    float cpu_time = has_schedstat ?
                     (float)schedstat.cpu_time_ns : 0.0f;
    features[FEAT_CPU_USAGE] =
        fminf(cpu_time / MAX_CPU_TIME_NS, 1.0f);

    /* Feature 1: I/O ratio */
    float io_bytes = has_io ?
        (float)(io.read_bytes + io.write_bytes) : 0.0f;
    features[FEAT_IO_RATIO] =
        fminf(io_bytes / MAX_IO_BYTES, 1.0f);

    /* Feature 2: Wait time (starvation) */
    float wait = has_schedstat ?
                 (float)schedstat.wait_time_ns : 0.0f;
    features[FEAT_WAIT_TIME] =
        fminf(wait / MAX_WAIT_TIME_NS, 1.0f);

    /* Feature 3: Burst length */
    float slices = has_schedstat ?
                   (float)schedstat.timeslices : 1.0f;
    float burst  = cpu_time / fmaxf(slices, 1.0f);
    features[FEAT_BURST_LEN] =
        fminf(burst / MAX_BURST_NS, 1.0f);

    /* Feature 4: Timeslice count */
    features[FEAT_TIMESLICES] =
        fminf(slices / MAX_TIMESLICES, 1.0f);

    /* Feature 5: Memory pressure */
    float mem = has_status ?
                (float)status.vm_rss_kb : 0.0f;
    features[FEAT_MEMORY] =
        fminf(mem / (float)total_ram_kb, 1.0f);

    /* Feature 6: Priority (normalised, clamped) */
    float prio = has_stat ? (float)stat.priority : 120.0f;
    features[FEAT_PRIORITY] =
        fmaxf(0.0f, fminf(prio / 139.0f, 1.0f));

    /* Feature 7: Thread count */
    float threads = has_stat ? (float)stat.threads : 1.0f;
    features[FEAT_THREADS] =
        fminf(threads / MAX_THREADS, 1.0f);

    /* Feature 8: State pattern from eBPF history */
    features[FEAT_STATE_PATTERN] =
        compute_state_pattern(hist);

    /* Feature 9: CPU affinity score */
    features[FEAT_CPU_AFFINITY] =
        compute_affinity(hist);

    return 0;
}

/* ── Debug printer ────────────────────────────────────────── */

void ml_print_features(int pid,
                       const float features[ML_INPUT_DIM])
{
    const char *names[ML_INPUT_DIM] = {
        "cpu_usage", "io_ratio",  "wait_time",
        "burst_len", "timeslices","memory",
        "priority",  "threads",   "state_pattern",
        "cpu_affinity"
    };

    printf("Features for PID %d:\n", pid);
    for (int i = 0; i < ML_INPUT_DIM; i++) {
        int bars = (int)(features[i] * 30);
        printf("  %15s: %.4f  ", names[i], features[i]);
        for (int b = 0; b < bars; b++) printf("█");
        printf("\n");
    }
}

/* ── Standalone test ──────────────────────────────────────── */
#ifdef ML_FEATURES_TEST
int main(int argc, char *argv[])
{
    int pid = (argc > 1) ? atoi(argv[1]) : getpid();

    printf("ML Feature Extractor — C Implementation\n");
    printf("========================================\n\n");

    struct pid_history hist = {0};
    float features[ML_INPUT_DIM];

    ml_extract_features(pid, &hist, features);
    ml_print_features(pid, features);

    /* Run inference on extracted features */
    printf("\nRunning inference...\n");
    struct ml_result r = ml_infer(features);

    const char *actions[] = {
        "LOWER_PRIO", "RAISE_PRIO", "KEEP", "BATCH", "NORMAL"
    };
    printf("Decision: %s (value=%.4f)\n",
           actions[r.action], r.value);

    return 0;
}
#endif
