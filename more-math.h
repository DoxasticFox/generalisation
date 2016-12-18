#ifndef MATH_H
#define MATH_H

float sgn(float x);
float clamp(float x, float min, float max);
float max(float x, float y);
float min(float x, float y);
float max(float *x, int n);
float min(float *x, int n);
void sub(float *x, int n, float c);
void div(float *x, int n, float c);
float mean(float* x, int n);
float var(float* x, int n);
void hadamard(float* x, float* y, int n);
float l2Norm(float* x, int n);

#endif
