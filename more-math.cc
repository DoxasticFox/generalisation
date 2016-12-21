#include "more-math.h"

float sgn(float x) {
  if (x > 0) return +1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

int clamp(int x, int min, int max) {
  if (x < min) return min;
  if (x > max) return max;
               return x;
}

float clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
               return x;
}

void maskedMoments(float *x, float *mask, int n, float &avg, float &std) {
  float avg_ = 0.0;
  float std_ = 0.0;

  // Compute average
  for (int i = 0; i < n; i++)
    if (mask[i])
      avg_ += x[i] / n;

  // Compute standard deviation
  for (int i = 0; i < n; i++) {
    if (mask[i]) {
      float d = (avg_ - x[i]);
      std_ += d * d / n;
    }
  }
  std_ = sqrt(std_);

  avg = avg_;
  std = std_;
}

float max(float x, float y) {
  if (x > y) return x;
  else       return y;
}

float min(float x, float y) {
  if (x < y) return x;
  else       return y;
}

float maskedAvg(float *x, bool *mask, int n) {
  float avg = 0.0;
  for (int i = 0; i < n; i++)
    if (mask[i])
      avg += x[i] / n;
  return avg;
}

float maskedMin(float *x, bool *mask, int n) {
  float best = FLT_MAX;
  for (int i = 0; i < n; i++)
    if (mask[i])
      if (x[i] < best) best = x[i];
  return best;
}

float maskedMax(float *x, bool *mask, int n) {
  float best = - FLT_MIN;
  for (int i = 0; i < n; i++)
    if (mask[i])
      if (x[i] > best) best = x[i];
  return best;
}

float min(float *x, int n) {
  float best = FLT_MAX;
  for (int i = 0; i < n; i++)
    if (x[i] < best) best = x[i];
  return best;
}

float max(float *x, int n) {
  float best = - FLT_MIN;
  for (int i = 0; i < n; i++)
    if (x[i] > best) best = x[i];
  return best;
}

void add(float *x, int n, float c) {
  for (int i = 0; i < n; i++) x[i] += c;
}

void mul(float *x, int n, float c) {
  for (int i = 0; i < n; i++) x[i] *= c;
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
