#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	if (x < 0) {
	    output[i] = -x;
	} else {
	    output[i] = x;
	}
    }
}

// implementation of absolute value using 15418 intrinsics
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
	//  reason: code doesn't deal with fontier situation.
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

		// All ones
		maskAll = _cmu418_init_ones();

		// All zeros
		maskIsNegative = _cmu418_init_ones(0);

		// Load vector of values from contiguous memory addresses
		_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

		// Set mask according to predicate, the first parameter is a quota.
		_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

		// Execute instruction using mask ("if" clause)
		_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

		// Inverse maskIsNegative to generate "else" mask
		maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

		// Execute instruction ("else" clause)
		_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

		// Write results back to memory
		_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// what is the meaning of "clamping value to 4.18"
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
		float x = values[i];	
		float result = 1.f;
		int y = exponents[i];
		float xpower = x;		
		while (y > 0) {
			// interesting bitewise tip:  0x1 is actually 0...1, then y & 0x1 is bitewise operation to test whether y is odd or even by comparing the last bite. here are two examples:
			// 0101 & 0001 == 1 (odd, true)
			// 0110 & 0001 == 0 (even, false)
			// this is a tricky method to caculate exponentiation by squaring. the reason of no (y-1) is that y >>= 1 will roundoff odd one.
			if (y & 0x1)
				result *= xpower;
			xpower = xpower * xpower;
			y >>= 1;
		}
		if (result > 4.18f) {
			result = 4.18f;
		}
		output[i] = result;
    }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // Implement your vectorized version of clampedExpSerial here
    //  ...

	fill(values+N, values+N+VECTOR_WIDTH, 0);
	__cmu418_vec_float x;
	__cmu418_vec_int y;
	__cmu418_vec_float xpower;
	__cmu418_vec_float result;
	__cmu418_vec_int zero = _cmu418_vset_int(0);
	__cmu418_vec_int one = _cmu418_vset_int(1);
	__cmu418_vec_float threashold;
	__cmu418_mask maskAll;

	for (int i=0; i<N+VECTOR_WIDTH; i+=VECTOR_WIDTH) {
		// these parameters should be defined in the for loop, or must be reclear at the every start of loop.
		__cmu418_mask  mask_greater_than_zero, mask_is_odd, mask_greater_than_threashold;

		bool b_is_last_seg = (i+1)>N ? true:false;
		int i_curr_vec_width = b_is_last_seg ? N%VECTOR_WIDTH : VECTOR_WIDTH;	// deal with the last division
		int i_offset = b_is_last_seg ? i-(VECTOR_WIDTH-i_curr_vec_width) : i;

		maskAll = _cmu418_init_ones();

		_cmu418_vload_float(x, values+i_offset, maskAll);						// float x = values[i];	
		result = _cmu418_vset_float(1.f);								// float result = 1.f;
		_cmu418_vload_int(y, exponents+i_offset, maskAll);						//  int y = exponents[i];
		_cmu418_vmove_float(xpower, x, maskAll);						//float xpower = x;		

		_cmu418_vgt_int(mask_greater_than_zero, y, zero, maskAll);
		int cntbits = _cmu418_cntbits(mask_greater_than_zero);
		while(cntbits != 0){
		// while (y > 0) { // exist y > 0
			__cmu418_vec_int tmp_mask_is_odd;
			_cmu418_vbitand_int(tmp_mask_is_odd, y, one, mask_greater_than_zero);

			// if (y & 0x1)
			_cmu418_vgt_int(mask_is_odd, tmp_mask_is_odd, zero, maskAll);
			__cmu418_mask tmp_odd_greater_mask = _cmu418_mask_and(mask_greater_than_zero, mask_is_odd);
			_cmu418_vmult_float(result, result, xpower, tmp_odd_greater_mask);
				// result *= xpower

			_cmu418_vmult_float(xpower, xpower, xpower, mask_greater_than_zero);
			// xpower = xpower * xpower;

			_cmu418_vshiftright_int(y, y, one, mask_greater_than_zero);
			// y >>= 1;

			_cmu418_vgt_int(mask_greater_than_zero, y, zero, maskAll);
			cntbits = _cmu418_cntbits(mask_greater_than_zero);
		}

		threashold = _cmu418_vset_float(4.18);
		_cmu418_vgt_float(mask_greater_than_threashold, result, threashold, maskAll);
		// if (result > 4.18f) 


		_cmu418_vset_float(result, 4.18f, mask_greater_than_threashold);
			// result = 4.18f;

		_cmu418_vstore_float(output+i_offset, result, maskAll);
		// output[i] = result;
    }
}


float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
		sum += values[i];
    }

    return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
    // Implement your vectorized version here
    //  ...
}
