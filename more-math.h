#ifndef MATH_H
#define MATH_H

int   pow2(int x);
float clamp(float x, float min, float max);
float max(float x, float y);
float mean(float* x, int n);
float var(float* x, int n);
void hadamard(float* x, float* y, int n);
float l2Norm(float* x, int n);

#endif
