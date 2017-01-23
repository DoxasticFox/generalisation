#include "unit.h"
#include "../unit.h"
#include "h.h"
#include <iostream>

enum Var { x, y, p };
enum Test { maximum, average };
static std::default_random_engine generator;

static void evalPerturbed(Unit &unit, double *inputs, double *outputs, double h, size_t size=0) {
  if (size == 0)
    size = unit.size;

  // Save original inputs;
  double *inputsPrev = new double[size];
  for (size_t i(0), I(size); i < I; i++) inputsPrev[i] = inputs[i];

  // Purturb inputs and evaluate
  for (size_t i(0), I(size); i < I; i++) inputs[i] += h;
  eval(unit);

  // Save outputs and restore original inputs
  for (size_t i(0), I(size); i < I; i++) outputs[i] = unit.zs[i];
  for (size_t i(0), I(size); i < I; i++) inputs [i] = inputsPrev[i];
}

static void initRandVec(double *&v, size_t n) {
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (size_t i(0); i < n; i++)
    v[i] = distribution(generator);
}

static void initOnesVec(double *&v, size_t n) {
  for (size_t i(0); i < n; i++) v[i] = 1.0;
}

static void makeRandVec(double *&v, size_t n) {
  v = new double[n];
  initRandVec(v, n);
}

static void makeOnesVec(double *&v, size_t n) {
  v = new double[n];
  initOnesVec(v, n);
}

static double errGradXYs(size_t n, Test test, Var v) {
  const size_t res = 20;
  const size_t gradSize = n;
  Unit unit = makeUnit(n, res);

  // Allocate and init vars
  double *xs, *ys, *numGrads, *zsSub, *zsAdd, *gradZs;
  makeRandVec(xs, gradSize);
  makeRandVec(ys, gradSize);
  numGrads = new double[gradSize];
  zsSub    = new double[gradSize];
  zsAdd    = new double[gradSize];
  makeOnesVec(gradZs, gradSize);

  // Choose variable to derive with respect to
  double *gradVar, *grad;
  if (v == Var::x) { gradVar = xs; grad = unit.gradXs; }
  if (v == Var::y) { gradVar = ys; grad = unit.gradYs; }

  // Compute zsSub, zsAdd
  setInputs(unit, xs, ys);
  evalPerturbed(unit, gradVar, zsSub, -h);
  evalPerturbed(unit, gradVar, zsAdd, +h);

  // Evaluate numerical gradient
  for (size_t i(0), I(gradSize); i < I; i++)
    numGrads[i] = (zsAdd[i] - zsSub[i]) / (2.0 * h);

  // Evaluate analytical gradients
  computeGrads(unit, gradZs);

  // Compute error
  long double err = 0.0;
  if (test == Test::average)
    for (size_t i(0), I(gradSize); i < I; i++)
      err += fabs(numGrads[i] - grad[i]) / ((long double) I);
  if (test == Test::maximum)
    for (size_t i(0), I(gradSize); i < I; i++)
      err = max(err, fabs(numGrads[i] - grad[i]));

  // Clean up
  delete[] xs;
  delete[] ys;
  delete[] numGrads;
  delete[] zsSub;
  delete[] zsAdd;
  delete[] gradZs;
  freeUnit(unit);

  return err;
}

static double errGradXs(size_t n, Test test) {
  return errGradXYs(n, test, Var::x);
}

static double errGradYs(size_t n, Test test) {
  return errGradXYs(n, test, Var::y);
}

static double errGradPs(size_t n, Test test) {
  const size_t size = 1;
  const size_t res = 3;
  const double reg = 0.0;
  const size_t gradSize = res * res;
  std::uniform_int_distribution<size_t> distribution(0, gradSize-1);

  long double err = 0.0;
  for (int i(0); i < n; i++) {
    Unit unit = makeUnit(size, res, reg);

    // Allocate and init vars
    double *xs, *ys, *numGrads, *zsSub, *zsAdd, *gradZs;
    makeRandVec(xs, unit.size);
    makeRandVec(ys, unit.size);
    numGrads = new double[gradSize];
    zsSub    = new double[unit.size];
    zsAdd    = new double[unit.size];
    makeOnesVec(gradZs, unit.size);

    // Choose variable to derive with respect to
    size_t gradVarIdx = distribution(generator);
    double *gradVar, *grad;
    gradVar = &unit.ps    [gradVarIdx];
    grad    = &unit.gradPs[gradVarIdx];

    // Compute zsSub, zsAdd
    setInputs(unit, xs, ys);
    evalPerturbed(unit, gradVar, zsSub, -h, 1);
    evalPerturbed(unit, gradVar, zsAdd, +h, 1);

    // Evaluate numerical gradient
    for (size_t i(0), I(unit.size); i < I; i++)
      numGrads[i] = (zsAdd[i] - zsSub[i]) / (2.0 * h);

    // Evaluate analytical gradients
    computeGrads(unit, gradZs);

    // Compute error
    if (test == Test::average)
      err += fabs(*numGrads - *grad) / ((long double) n);
    if (test == Test::maximum)
      err = max(err, fabs(*numGrads - *grad));

    // Clean up
    delete[] xs;
    delete[] ys;
    delete[] numGrads;
    delete[] zsSub;
    delete[] zsAdd;
    delete[] gradZs;
    freeUnit(unit);
  }

  return err;
}

static double errGrad(size_t n, Var v, Test test) {
  if (v == Var::x) return errGradXs(n, test);
  if (v == Var::y) return errGradYs(n, test);
  if (v == Var::p) return errGradPs(n, test);
  return -1.0;
}

static double errAct() {
  const size_t size = 1;
  const size_t res  = 3;
  Unit unit = makeUnit(size, res);

  double p11, p21, p31,
         p12, p22, p32,
         p13, p23, p33;

  p11 = unit.ps[0] = 0.8;
  p21 = unit.ps[1] = 0.5;
  p31 = unit.ps[2] = 0.3;
  p12 = unit.ps[3] = 0.4;
  p22 = unit.ps[4] = 0.6;
  p32 = unit.ps[5] = 0.1;
  p13 = unit.ps[6] = 0.7;
  p23 = unit.ps[7] = 0.2;
  p33 = unit.ps[8] = 0.9;

  double xy[] = {};


  return -1.0;
}

void testUnit() {
  size_t n = 10000;

  std::cout << "Test: unit"                                      << std::endl;
  std::cout << "  avg(x): " << errGrad(n, Var::x, Test::average) << std::endl;
  std::cout << "  max(x): " << errGrad(n, Var::x, Test::maximum) << std::endl;
  std::cout << "  avg(y): " << errGrad(n, Var::y, Test::average) << std::endl;
  std::cout << "  max(y): " << errGrad(n, Var::y, Test::maximum) << std::endl;
  std::cout << "  avg(p): " << errGrad(n, Var::p, Test::average) << std::endl;
  std::cout << "  max(p): " << errGrad(n, Var::p, Test::maximum) << std::endl;
  std::cout << "       z: " << errAct ()                         << std::endl;
}
