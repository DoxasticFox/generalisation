#include "blerp.h"
#include "../blerp.h"
#include "h.h"
#include <iostream>

enum Var { x, y, p11, p12, p21, p22 };
static double errGrad(int n,Var v);
static double numGradP22(double x,double y,double x1,double x2,double y1,double y2,double p11,double p12,double p21,double p22);
static double numGradP21(double x,double y,double x1,double x2,double y1,double y2,double p11,double p12,double p21,double p22);
static double numGradP12(double x,double y,double x1,double x2,double y1,double y2,double p11,double p12,double p21,double p22);
static double numGradP11(double x,double y,double x1,double x2,double y1,double y2,double p11,double p12,double p21,double p22);
static double numGradY(double x,double y,double x1,double x2,double y1,double y2,double p11,double p12,double p21,double p22);
static double numGradX(double x,double y,double x1,double x2,double y1,double y2,double p11,double p12,double p21,double p22);

static double numGradX(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  double fSub = blerp(x-h, y, x1, x2, y1, y2, p11, p12, p21, p22);
  double fAdd = blerp(x+h, y, x1, x2, y1, y2, p11, p12, p21, p22);

  return (fAdd - fSub) / (2.0 * h);
}

static double numGradY(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  double fSub = blerp(x, y-h, x1, x2, y1, y2, p11, p12, p21, p22);
  double fAdd = blerp(x, y+h, x1, x2, y1, y2, p11, p12, p21, p22);

  return (fAdd - fSub) / (2.0 * h);
}

static double numGradP11(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  double fSub = blerp(x, y, x1, x2, y1, y2, p11-h, p12, p21, p22);
  double fAdd = blerp(x, y, x1, x2, y1, y2, p11+h, p12, p21, p22);

  return (fAdd - fSub) / (2.0 * h);
}

static double numGradP12(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  double fSub = blerp(x, y, x1, x2, y1, y2, p11, p12-h, p21, p22);
  double fAdd = blerp(x, y, x1, x2, y1, y2, p11, p12+h, p21, p22);

  return (fAdd - fSub) / (2.0 * h);
}

static double numGradP21(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  double fSub = blerp(x, y, x1, x2, y1, y2, p11, p12, p21-h, p22);
  double fAdd = blerp(x, y, x1, x2, y1, y2, p11, p12, p21+h, p22);

  return (fAdd - fSub) / (2.0 * h);
}

static double numGradP22(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  double fSub = blerp(x, y, x1, x2, y1, y2, p11, p12, p21, p22-h);
  double fAdd = blerp(x, y, x1, x2, y1, y2, p11, p12, p21, p22+h);

  return (fAdd - fSub) / (2.0 * h);
}

double errGrad(int n, Var v) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  long double diffSum = 0.0;

  for (int i = 0; i < n; i++) {
    double x   = distribution(generator);
    double y   = distribution(generator);
    double x1  = distribution(generator) / 2.0;
    double x2  = distribution(generator) / 2.0 + x1 + 1e-5;
    double y1  = distribution(generator) / 2.0;
    double y2  = distribution(generator) / 2.0 + y1 + 1e-5;
    double p11 = distribution(generator);
    double p12 = distribution(generator);
    double p21 = distribution(generator);
    double p22 = distribution(generator);

    double fun = 0.0;
    double num = 0.0;

    if (v == Var::x) {
      fun = gradBlerpX(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
      num =   numGradX(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
    }
    if (v == Var::y) {
      fun = gradBlerpY(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
      num =   numGradY(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
    }
    if (v == Var::p11) {
      fun = gradBlerpP11(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
      num =   numGradP11(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
    }
    if (v == Var::p12) {
      fun = gradBlerpP12(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
      num =   numGradP12(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
    }
    if (v == Var::p21) {
      fun = gradBlerpP21(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
      num =   numGradP21(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
    }
    if (v == Var::p22) {
      fun = gradBlerpP22(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
      num =   numGradP22(x, y, x1, x2, y1, y2, p11, p12, p21, p22);
    }

    diffSum += fabs(fun - num);
  }

  return diffSum / (long double) n;
}

void testBlerp() {
  int n = 10000;

  std::cout << "Test: blerp"                << std::endl;
  std::cout << "    x: " << errGrad(n, Var::x  ) << std::endl;
  std::cout << "    y: " << errGrad(n, Var::y  ) << std::endl;
  std::cout << "  p11: " << errGrad(n, Var::p11) << std::endl;
  std::cout << "  p12: " << errGrad(n, Var::p12) << std::endl;
  std::cout << "  p21: " << errGrad(n, Var::p21) << std::endl;
  std::cout << "  p22: " << errGrad(n, Var::p22) << std::endl;
}
