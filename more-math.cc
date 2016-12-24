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

int count(bool *x, int n) {
  int c = 0;
  for (int i = 0; i < n; i++)
    if (x[i]) c++;
  return c;
}

void maskedMoments(float *x, bool *mask, int n, float &avg, float &var) {
  float avg_ = 0.0;
  float var_ = 0.0;
  float cnt_ = count(mask, n);

  // Compute average
  for (int i = 0; i < n; i++)
    if (mask[i])
      avg_ += x[i] / cnt_;

  // Compute variance
  for (int i = 0; i < n; i++) {
    if (mask[i]) {
      float d = (avg_ - x[i]);
      var_ += d * d / cnt_;
    }
  }

  avg = avg_;
  var = var_;
}

float maskedAvg(float *x, bool *mask, int n) {
  float cnt = count(mask, n);
  float avg = 0.0;
  for (int i = 0; i < n; i++)
    if (mask[i])
      avg += x[i] / cnt;
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

void moments(float *x, int n, float &avg, float &var) {
  float avg_ = 0.0;
  float var_ = 0.0;

  // Compute average
  for (int i = 0; i < n; i++)
    avg_ += x[i] / n;

  // Compute variance
  for (int i = 0; i < n; i++) {
    float d = (avg_ - x[i]);
    var_ += d * d / n;
  }

  avg = avg_;
  var = var_;
}

float max(float x, float y) {
  if (x > y) return x;
  else       return y;
}

float min(float x, float y) {
  if (x < y) return x;
  else       return y;
}

float max(float *x, int n) {
  float best = - FLT_MIN;
  for (int i = 0; i < n; i++)
    if (x[i] > best) best = x[i];
  return best;
}

float min(float *x, int n) {
  float best = FLT_MAX;
  for (int i = 0; i < n; i++)
    if (x[i] < best) best = x[i];
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

float var(float* x, int n) {
  float m = mean(x, n);
  float var = 0.0;
  for (int i = 0; i < n; i++) {
    float d = x[i] - m;
    var += d * d / n;
  }
  return var;
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
