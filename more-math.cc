#include <math.h>
#include "math.h"

float clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
               return x;
}

float max(float x, float y) {
  if (x > y) return x;
  else       return y;
}

float mean(float* x, int n) {
  float m = 0.0;
  for (int i = 0; i < n; i++)
    m += x[i] / n;
  return m;
}

// Is this the standard deviation, or variance?
float var(float* x, int n) {
  float m = mean(x, n);
  float std = 0.0;
  for (int i = 0; i < n; i++) {
    float d = x[i] - m;
    std += d * d / n;
  }
  return std;
}

void hadamard(float* x, float* y, int n) {
  for (int i = 0; i < n; i++)
    x[i] *= y[i];
}

float l2Norm(float* x, int n) {
  float norm = 0.0;
  for (int i = 0; i < n; i++)
    norm += x[i] * x[i];
  norm = sqrt(norm);
  return norm;
}
