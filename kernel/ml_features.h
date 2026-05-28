/*
 * ml_features.h
 * Reads /proc filesystem and produces ML feature vectors.
 * Pure C — no Python, no dependencies.
 */

#ifndef ML_FEATURES_H
#define ML_FEATURES_H

#include "ml_inference.h"

/* Feature vector indices — matches Python extractor exactly */
#define FEAT_CPU_USAGE     0
#define FEAT_IO_RATIO      1
#define FEAT_WAIT_TIME     2
#define FEAT_BURST_LEN     3
#define FEAT_TIMESLICES    4
#define FEAT_MEMORY        5
#define FEAT_PRIORITY      6
#define FEAT_THREADS       7
#define FEAT_STATE_PATTERN 8
#define FEAT_CPU_AFFINITY  9

/* Normalisation constants — must match Python extractor */
#define MAX_CPU_TIME_NS    100000000000.0f
#define MAX_WAIT_TIME_NS   100000000000.0f
#define MAX_TIMESLICES     1000000.0f
#define MAX_IO_BYTES       10000000000.0f
#define MAX_THREADS        1000.0f
#define MAX_BURST_NS       10000000.0f

/* Per-process state history for state_pattern feature */
#define HISTORY_LEN        50

struct pid_history {
    int   pid;
    int   states[HISTORY_LEN];   /* prev_state ring buffer */
    int   cpus[HISTORY_LEN];     /* cpu_id ring buffer */
    int   head;                  /* current write position */
    int   count;                 /* how many entries filled */
};

/* Extract 10-element feature vector for a given PID */
int ml_extract_features(int pid,
                        struct pid_history *hist,
                        float features[ML_INPUT_DIM]);

/* Update history with a new scheduling event */
void ml_history_update(struct pid_history *hist,
                       int pid, int state, int cpu);

/* Print feature vector for debugging */
void ml_print_features(int pid,
                       const float features[ML_INPUT_DIM]);

#endif /* ML_FEATURES_H */
