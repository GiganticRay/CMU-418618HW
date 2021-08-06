#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
    N:  iteration number
    initialGuess: just as the name
*/
void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    // error tolerance threshold.
    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        // NewTon's iterative method. guess ≈ sqrt(1/value). output ≈ value * sqrt(1/value) = sqrt(value).
        // the region of x is (0, 3)
        // formula infer: using newton formula X_(n+1) = X_n - f(x_n)/f'(x_n) to iteratively get the solution of f(x) = 0. the formula below is the iterative process of f(x) = 1/x^2 - s = 0.
        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

