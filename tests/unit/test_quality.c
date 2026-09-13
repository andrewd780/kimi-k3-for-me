#include "k3_quality.h"
#include <stdio.h>

int main(void)
{
    double loss;
    const float uniform[] = {0, 0, 0, 0};
    const float shifted[] = {1e20f, 1e20f, 1e20f, 1e20f};
    const float separated[] = {1000, -1000};
    const float nan_input[] = {0, NAN}, inf_input[] = {0, INFINITY};
    if (k3_token_nll(uniform, 4, 2, &loss) || fabs(loss - log(4.0)) > 1e-15 ||
        k3_token_nll(shifted, 4, 1, &loss) || fabs(loss - log(4.0)) > 1e-15 ||
        k3_token_nll(separated, 2, 1, &loss) || loss != 2000.0 ||
        k3_token_nll(separated, 2, 0, &loss) || loss != 0.0 ||
        !k3_token_nll(nan_input, 2, 0, &loss) || !k3_token_nll(inf_input, 2, 0, &loss) ||
        !k3_token_nll(uniform, 4, 4, &loss) || !k3_token_nll(uniform, 0, 0, &loss)) {
        fprintf(stderr, "quality arithmetic failed\n");
        return 1;
    }
    puts("QUALITY ARITHMETIC PASSED (synthetic logits only)");
    return 0;
}
