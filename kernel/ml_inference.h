/*
 * ml_inference.h
 * Tiny ML inference engine for the quantum scheduler.
 * No dependencies — pure C, runs in kernel or userspace.
 */

#ifndef ML_INFERENCE_H
#define ML_INFERENCE_H

#define ML_INPUT_DIM   10
#define ML_HIDDEN_DIM  64
#define ML_ACTION_DIM   5

/* Scheduling actions */
#define ACTION_LOWER_PRIO    0
#define ACTION_RAISE_PRIO    1
#define ACTION_KEEP          2
#define ACTION_BATCH         3
#define ACTION_NORMAL        4

/* Result of one inference pass */
struct ml_result {
    int   action;                      /* chosen action 0-4 */
    float probs[ML_ACTION_DIM];        /* probability per action */
    float value;                       /* critic value estimate */
};

/* Main inference function */
struct ml_result ml_infer(const float features[ML_INPUT_DIM]);

/* Verify inference engine matches Python output */
int ml_verify(void);

#endif /* ML_INFERENCE_H */
