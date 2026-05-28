/*
 * ml_inference.c
 * Pure C implementation of the Actor-Critic network.
 * Matches exactly: 10 → 64 → 64 → [actor:5, critic:1]
 *
 * Verification target (from Python export):
 *   Input:  [0.06, 0.04, 0.007, 0.09, 0.006,
 *            0.06, 0.14, 0.08,  0.06, 1.0]
 *   Probs:  [0.1822, 0.1791, 0.2865, 0.1486, 0.2037]
 *   Value:  2.538386
 *   Action: 2
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "ml_inference.h"
#include "ml_weights.h"

/* ── Math helpers ─────────────────────────────────────────── */

static float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

/*
 * Linear layer forward pass.
 * out[i] = bias[i] + sum(weight[i*in_dim + j] * in[j])
 * weight stored row-major: row i = weights for output neuron i
 */
static void linear(const float *in,   int in_dim,
                   const float *w,    const float *b,
                   float       *out,  int out_dim)
{
    for (int i = 0; i < out_dim; i++) {
        float sum = b[i];
        for (int j = 0; j < in_dim; j++)
            sum += w[i * in_dim + j] * in[j];
        out[i] = sum;
    }
}

/*
 * ReLU applied in-place across an array.
 */
static void relu_vec(float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = relu(x[i]);
}

/*
 * Softmax — converts raw logits to probabilities.
 * Numerically stable: subtract max before exp.
 */
static void softmax(const float *logits, float *probs, int n)
{
    /* Find max for numerical stability */
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val)
            max_val = logits[i];

    /* Exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    /* Normalize */
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

/*
 * Argmax — returns index of highest probability.
 */
static int argmax(const float *probs, int n)
{
    int   best_idx = 0;
    float best_val = probs[0];
    for (int i = 1; i < n; i++) {
        if (probs[i] > best_val) {
            best_val = probs[i];
            best_idx = i;
        }
    }
    return best_idx;
}

/* ── Main inference ───────────────────────────────────────── */

struct ml_result ml_infer(const float features[ML_INPUT_DIM])
{
    struct ml_result result;

    /* Buffers for intermediate activations */
    float h1[ML_HIDDEN_DIM];   /* after layer 1 */
    float h2[ML_HIDDEN_DIM];   /* after layer 2 */
    float actor_logits[ML_ACTION_DIM];
    float critic_out[1];

    /* ── Shared encoder ─────────────────────────────────── */

    /* Layer 1: 10 → 64 */
    linear(features,   ML_INPUT_DIM,
           shared_0_weight, shared_0_bias,
           h1, ML_HIDDEN_DIM);
    relu_vec(h1, ML_HIDDEN_DIM);

    /* Layer 2: 64 → 64 */
    linear(h1,         ML_HIDDEN_DIM,
           shared_2_weight, shared_2_bias,
           h2, ML_HIDDEN_DIM);
    relu_vec(h2, ML_HIDDEN_DIM);

    /* ── Actor head: 64 → 5 ─────────────────────────────── */
    linear(h2,         ML_HIDDEN_DIM,
           actor_weight, actor_bias,
           actor_logits, ML_ACTION_DIM);

    softmax(actor_logits, result.probs, ML_ACTION_DIM);
    result.action = argmax(result.probs, ML_ACTION_DIM);

    /* ── Critic head: 64 → 1 ────────────────────────────── */
    linear(h2,         ML_HIDDEN_DIM,
           critic_weight, critic_bias,
           critic_out, 1);
    result.value = critic_out[0];

    return result;
}

/* ── Verification ─────────────────────────────────────────── */

int ml_verify(void)
{
    /* Same test input used during Python export */
    const float test_input[ML_INPUT_DIM] = {
        0.06f, 0.04f, 0.007f, 0.09f, 0.006f,
        0.06f, 0.14f, 0.08f,  0.06f, 1.0f
    };

    /* Expected output from Python */
    const float expected_probs[ML_ACTION_DIM] = {
        0.1822f, 0.1791f, 0.2865f, 0.1486f, 0.2037f
    };
    const float expected_value  = 2.538386f;
    const int   expected_action = 2;
    const float tolerance       = 0.001f;  /* allow 0.1% float error */

    struct ml_result r = ml_infer(test_input);

    printf("C Inference verification:\n");
    printf("  Action probs: [");
    for (int i = 0; i < ML_ACTION_DIM; i++)
        printf("%.4f%s", r.probs[i],
               i < ML_ACTION_DIM-1 ? ", " : "");
    printf("]\n");
    printf("  Value:  %.6f\n", r.value);
    printf("  Action: %d\n",   r.action);

    /* Check each probability */
    int pass = 1;
    for (int i = 0; i < ML_ACTION_DIM; i++) {
        float diff = fabsf(r.probs[i] - expected_probs[i]);
        if (diff > tolerance) {
            printf("  FAIL: prob[%d] = %.4f, expected %.4f "
                   "(diff %.4f)\n",
                   i, r.probs[i], expected_probs[i], diff);
            pass = 0;
        }
    }

    float val_diff = fabsf(r.value - expected_value);
    if (val_diff > tolerance) {
        printf("  FAIL: value = %.6f, expected %.6f\n",
               r.value, expected_value);
        pass = 0;
    }

    if (r.action != expected_action) {
        printf("  FAIL: action = %d, expected %d\n",
               r.action, expected_action);
        pass = 0;
    }

    if (pass)
        printf("  ✓ All checks passed — C matches Python exactly\n");
    else
        printf("  ✗ Verification FAILED — check weight export\n");

    return pass;
}

/* ── Test main ────────────────────────────────────────────── */
#ifdef ML_STANDALONE_TEST
int main(void)
{
    printf("ML Scheduler Inference Engine\n");
    printf("=============================\n\n");

    /* Run verification */
    int ok = ml_verify();

    /* Demo inference on a custom input */
    printf("\nDemo inference:\n");
    float demo[ML_INPUT_DIM] = {
        0.95f,  /* cpu_usage   — very CPU hungry */
        0.01f,  /* io_ratio    — not I/O bound */
        0.80f,  /* wait_time   — being starved */
        0.90f,  /* burst_len   — long bursts */
        0.50f,  /* timeslices  — moderate frequency */
        0.20f,  /* memory      — moderate memory */
        0.50f,  /* priority    — average priority */
        0.01f,  /* threads     — single threaded */
        0.95f,  /* state       — always CPU bound */
        1.00f,  /* affinity    — stays on one core */
    };

    struct ml_result r = ml_infer(demo);

    const char *action_names[] = {
        "LOWER_PRIO", "RAISE_PRIO", "KEEP",
        "BATCH", "NORMAL"
    };

    printf("  Input: CPU-hungry, starved, CPU-bound process\n");
    printf("  Action: %s (expected: RAISE_PRIO or BATCH)\n",
           action_names[r.action]);
    printf("  Value:  %.4f\n", r.value);
    printf("  Probs:  ");
    for (int i = 0; i < ML_ACTION_DIM; i++)
        printf("%s=%.3f ", action_names[i], r.probs[i]);
    printf("\n");

    return ok ? 0 : 1;
}
#endif
