#ifndef MORE_MATH_H
#define MORE_MATH_H

#include <float.h>
#include <math.h>

float sgn(float x);
int clamp(int x, int min, int max);
float clamp(float x, float min, float max);
void maskedMoments(float *x, float *mask, int n, float &avg, float &std);
float max(float x, float y);
float min(float x, float y);
float maskedAvg(float *x, bool *mask, int n);
float maskedMin(float *x, bool *mask, int n);
float maskedMax(float *x, bool *mask, int n);
float max(float *x, int n);
float min(float *x, int n);
void add(float *x, int n, float c);
void mul(float *x, int n, float c);
float mean(float* x, int n);
float var(float* x, int n);
void hadamard(float* x, float* y, int n);
float l2Norm(float* x, int n);

#endif
