#include "more-math.h"

double sgn(double x) {
  if (x > 0) return +1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

int clamp(ptrdiff_t x, ptrdiff_t min, ptrdiff_t max) {
  if (x < min) return min;
  if (x > max) return max;
               return x;
}

int clamp(int x, int min, int max) {
  if (x < min) return min;
  if (x > max) return max;
               return x;
}

double clamp(double x, double min, double max) {
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

void maskedMoments(double *x, bool *mask, int n, double &avg, double &var) {
  double avg_ = 0.0;
  double var_ = 0.0;
  double cnt_ = count(mask, n);

  // Compute average
  for (int i = 0; i < n; i++)
    if (mask[i])
      avg_ += x[i] / cnt_;

  // Compute variance
  for (int i = 0; i < n; i++) {
    if (mask[i]) {
      double d = (avg_ - x[i]);
      var_ += d * d / cnt_;
    }
  }

  avg = avg_;
  var = var_;
}

double maskedAvg(double *x, bool *mask, int n) {
  double cnt = count(mask, n);
  double avg = 0.0;
  for (int i = 0; i < n; i++)
    if (mask[i])
      avg += x[i] / cnt;
  return avg;
}

double maskedMin(double *x, bool *mask, int n) {
  double best = DBL_MAX;
  for (int i = 0; i < n; i++)
    if (mask[i])
      if (x[i] < best) best = x[i];
  return best;
}

double maskedMax(double *x, bool *mask, int n) {
  double best = - DBL_MIN;
  for (int i = 0; i < n; i++)
    if (mask[i])
      if (x[i] > best) best = x[i];
  return best;
}

void moments(double *x, int n, double &avg, double &var) {
  double avg_ = 0.0;
  double var_ = 0.0;

  // Compute average
  for (int i = 0; i < n; i++)
    avg_ += x[i] / n;

  // Compute variance
  for (int i = 0; i < n; i++) {
    double d = (avg_ - x[i]);
    var_ += d * d / n;
  }

  avg = avg_;
  var = var_;
}

double max(double x, double y) {
  if (x > y) return x;
  else       return y;
}

double min(double x, double y) {
  if (x < y) return x;
  else       return y;
}

double max(double *x, int n) {
  double best = - DBL_MIN;
  for (int i = 0; i < n; i++)
    if (x[i] > best) best = x[i];
  return best;
}

double min(double *x, int n) {
  double best = DBL_MAX;
  for (int i = 0; i < n; i++)
    if (x[i] < best) best = x[i];
  return best;
}

void add(double *x, int n, double c) {
  for (int i = 0; i < n; i++) x[i] += c;
}

void mul(double *x, int n, double c) {
  for (int i = 0; i < n; i++) x[i] *= c;
}

double mean(double* x, int n) {
  double m = 0.0;
  for (int i = 0; i < n; i++)
    m += x[i] / n;
  return m;
}

double var(double* x, int n) {
  double m = mean(x, n);
  double var = 0.0;
  for (int i = 0; i < n; i++) {
    double d = x[i] - m;
    var += d * d / n;
  }
  return var;
}

void hadamard(double* x, double* y, int n) {
  for (int i = 0; i < n; i++)
    x[i] *= y[i];
}

double l2Norm(double* x, int n) {
  double norm = 0.0;
  for (int i = 0; i < n; i++)
    norm += x[i] * x[i];
  norm = sqrt(norm);
  return norm;
}
