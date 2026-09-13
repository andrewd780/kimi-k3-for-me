/* Evaluation arithmetic only. No approximation or change to inference logits. */
#ifndef K3_QUALITY_H
#define K3_QUALITY_H
#include <math.h>

static inline int k3_token_nll(const float *logits, int vocab, int target, double *nll)
{
    if (!logits || !nll || vocab <= 0 || target < 0 || target >= vocab) return -1;
    double maximum = logits[0];
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) return -1;
        if ((double)logits[i] > maximum) maximum = logits[i];
    }
    double sum = 0.0;
    for (int i = 0; i < vocab; i++) sum += exp((double)logits[i] - maximum);
    /* Subtract before adding log(sum), retaining small losses at large logit offsets. */
    *nll = log(sum) + (maximum - (double)logits[target]);
    return isfinite(*nll) && *nll >= 0.0 ? 0 : -1;
}
#endif
