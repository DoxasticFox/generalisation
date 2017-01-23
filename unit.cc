#include "unit.h"

/* Indexing function for `Unit.ps`, which represents a 2-d grid of control
 * points.
 */
ptrdiff_t P(size_t res, ptrdiff_t x, ptrdiff_t y) { return x + res * y; }

Unit makeUnit(size_t size, size_t res, double reg, bool doBatchNorm) {
  Unit unit;

  setSize     (unit, size);
  setRes      (unit, res);
  setReg      (unit, reg);
  setBatchNorm(unit, doBatchNorm);

  initUnit    (unit);

  return unit;
}

void freeUnit(Unit &unit) {
  free(unit.zs);
  free(unit.gradXs);
  free(unit.gradYs);
  free(unit.bins);
  free(unit.pIdxs);
  free(unit.pLocs);
  free(unit.pVals);
  free(unit.gradPsEx);
  free(unit.ps);
  free(unit.gradPs);
  free(unit.gradPsReg);
  free(unit.gradPsRegX);
}

void initUnit(Unit &unit) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (size_t x(0), X(unit.res); x < X; x++) {
    for (size_t y(0), Y(unit.res); y < Y; y++) {
      unit.ps[P(X, x, y)] = distribution(generator);

      //double fx = x / (X - 1.0);
      //double fy = y / (Y - 1.0);
      //
      //if ((fx > 0.5) != (fy > 0.5)) unit.ps[P(X, x, y)] = 0.0;
      //else                          unit.ps[P(X, x, y)] = 1.0;
      //
      //unit.ps[P(X, x, y)] = (fx + fy) / 2.0;
    }
  }
}

void setSize(Unit &unit, size_t size) {
  unit.zs       = (double*) realloc(unit.zs,        sizeof(double) * size);
  unit.gradXs   = (double*) realloc(unit.gradXs,    sizeof(double) * size);
  unit.gradYs   = (double*) realloc(unit.gradYs,    sizeof(double) * size);
  unit.bins     = (Bins*)   realloc(unit.bins,      sizeof(Bins)   * size);
  unit.pIdxs    = (PIdxs*)  realloc(unit.pIdxs,     sizeof(PIdxs)  * size);
  unit.pLocs    = (PLocs*)  realloc(unit.pLocs,     sizeof(PLocs)  * size);
  unit.pVals    = (PVals*)  realloc(unit.pVals,     sizeof(PVals)  * size);
  unit.gradPsEx = (PGrads*) realloc(unit.gradPsEx,  sizeof(PGrads) * size);

  unit.size = size;
}

void setRes(Unit &unit, size_t res) {
  size_t numPs = res * res;

  unit.ps         = (double*) realloc(unit.ps,         sizeof(double) * numPs);
  unit.gradPs     = (double*) realloc(unit.gradPs,     sizeof(double) * numPs);
  unit.gradPsReg  = (double*) realloc(unit.gradPsReg,  sizeof(double) * numPs);
  unit.gradPsRegX = (double*) realloc(unit.gradPsRegX, sizeof(double) * numPs);

  unit.res = res;
}

void setReg(Unit &unit, double reg) {
  unit.reg = reg;
}

void setBatchNorm(Unit &unit, bool doBatchNorm) {
  unit.doBatchNorm = doBatchNorm;
}

void setGradZs(Unit &unit, double *gradZs) {
  unit.gradZs = gradZs;
}

void setXs(Unit &unit, double *xs) { unit.xs = xs; }
void setYs(Unit &unit, double *ys) { unit.ys = ys; }
void setInputs(Unit &unit, double *xs, double *ys) {
  setXs(unit, xs);
  setYs(unit, ys);
}

void eval(Unit &unit, double *xs, double *ys, bool doBatchNorm) {
  setInputs(unit, xs, ys);
  setBatchNorm(unit, doBatchNorm);

  eval(unit);
}

void eval(Unit &unit) {
  computeBins (unit);
  computePIdxs(unit);
  computePLocs(unit);
  computePVals(unit);
  computeZs(unit);

  if (unit.doBatchNorm)
    batchNorm(unit);
}

void computeBins(Unit &unit) {
  for (size_t i(0), I(unit.size), res(unit.res); i < I; i++)
    unit.bins[i].x = bin(unit.xs[i], res);
  for (size_t i(0), I(unit.size), res(unit.res); i < I; i++)
    unit.bins[i].y = bin(unit.ys[i], res);
}

ptrdiff_t bin(double x, size_t res) {
  ptrdiff_t b;
  b = (ptrdiff_t) (x * (res - 1));
  b = clamp(b, (ptrdiff_t) 0, (ptrdiff_t) (res - 2));
  return b;
}

void computePIdxs(Unit &unit) {
  for (size_t i(0), I(unit.size), res(unit.res); i < I; i++)
    computePIdxs(unit.bins[i], unit.pIdxs[i], res);
}

void computePIdxs(Bins &bins, PIdxs &pIdxs, size_t res) {
  pIdxs.p11 = P(res, bins.x, bins.y);
  pIdxs.p21 = pIdxs.p11 + 1;
  pIdxs.p12 = pIdxs.p11 + res;
  pIdxs.p22 = pIdxs.p12 + 1;
}

void computePLocs(Unit &unit) {
  for (size_t i(0), I(unit.size), res(unit.res); i < I; i++)
    computePLocs(unit.xs[i], unit.ys[i], unit.pLocs[i], res);
}

void computePLocs(double x, double y, PLocs &pLocs, size_t res) {
  double numGaps = (res - 1.0);
  double pitch = 1.0 / numGaps; // Distance between control points

  pLocs.x1 = bin(x, res) / numGaps;
  pLocs.y1 = bin(y, res) / numGaps;
  pLocs.x2 = pLocs.x1 + pitch;
  pLocs.y2 = pLocs.y1 + pitch;
}

void computePVals(Unit &unit) {
  for (size_t i(0), I(unit.size); i < I; i++)
    computePVals(unit.ps, unit.pIdxs[i], unit.pVals[i]);
}

void computePVals(double *ps, PIdxs &pIdxs, PVals &pVals) {
  pVals.p11 = ps[pIdxs.p11];
  pVals.p12 = ps[pIdxs.p12];
  pVals.p21 = ps[pIdxs.p21];
  pVals.p22 = ps[pIdxs.p22];
}

void computeZs(Unit &unit) {
  for (size_t i(0), I(unit.size); i < I; i++)
    unit.zs[i] = blerp(unit.xs[i], unit.ys[i], unit.pLocs[i], unit.pVals[i]);
  for (size_t i(0), I(unit.size); i < I; i++)
    unit.zs[i] = clamp(unit.zs[i], 0.0, 1.0);
}

double blerp(double x, double y, PLocs &pLocs, PVals &pVals) {
  return blerp(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

/**************************** GRADIENT COMPUTATION ****************************/

void computeGrads(Unit &unit, double *gradZs) {
  setGradZs(unit, gradZs);

  computeGrads(unit);
}

void computeGrads(Unit &unit) {
  computeXGrads(unit);
  computeYGrads(unit);
  computePGrads(unit);
}

void computeXGrads(Unit &unit) {
  for (size_t i(0), I(unit.size); i < I; i++) {
    double xs    = unit.xs[i];
    double ys    = unit.ys[i];
    PLocs pLocs = unit.pLocs[i];
    PVals pVals = unit.pVals[i];
    double gradX = computeXGrads(xs, ys, pLocs, pVals);

    unit.gradXs[i] = gradX;
  }

  // Gradient due to clamping
  for (size_t i(0), I(unit.size); i < I; i++)
    if (unit.zs[i] <= 0.0 || unit.zs[i] >= 1.0)
      unit.gradXs[i] = 0.0;

  // Gradient due to back-prop
  for (size_t i(0), I(unit.size); i < I; i++)
    unit.gradXs[i] *= unit.gradZs[i];
}

void computeYGrads(Unit &unit) {
  for (size_t i(0), I(unit.size); i < I; i++) {
    double xs    = unit.xs[i];
    double ys    = unit.ys[i];
    PLocs pLocs = unit.pLocs[i];
    PVals pVals = unit.pVals[i];
    double gradY = computeYGrads(xs, ys, pLocs, pVals);

    unit.gradYs[i] = gradY;
  }

  // Gradient due to clamping
  for (size_t i(0), I(unit.size); i < I; i++)
    if (unit.zs[i] <= 0.0 || unit.zs[i] >= 1.0)
      unit.gradYs[i] = 0.0;

  // Gradient due to back-prop
  for (size_t i(0), I(unit.size); i < I; i++)
    unit.gradYs[i] *= unit.gradZs[i];
}

void computePGrads(Unit &unit) {
  computeExPGrads (unit);
  computeRegPGrads(unit);

  // Add/copy regularisation grads
  for (int i(0), I(unit.res * unit.res); i < I; i++)
    unit.gradPs[i] = unit.gradPsReg[i];

  // Add example grads
  for (int i(0), I(unit.size); i < I; i++) {
    unit.gradPs[unit.pIdxs[i].p11] += unit.gradPsEx[i].p11 / ((double) I);
    unit.gradPs[unit.pIdxs[i].p12] += unit.gradPsEx[i].p12 / ((double) I);
    unit.gradPs[unit.pIdxs[i].p21] += unit.gradPsEx[i].p21 / ((double) I);
    unit.gradPs[unit.pIdxs[i].p22] += unit.gradPsEx[i].p22 / ((double) I);
  }
}

/*
 * Gradients of output with respect to control points
 */
void computeExPGrads(Unit &unit) {
  for (size_t i(0), I(unit.size); i < I; i++) {
    unit.gradPsEx[i].p11 = computeP11Grads(
        unit.xs[i], unit.ys[i],
        unit.pLocs[i],
        unit.pVals[i]
    );
    unit.gradPsEx[i].p12 = computeP12Grads(
        unit.xs[i], unit.ys[i],
        unit.pLocs[i],
        unit.pVals[i]
    );
    unit.gradPsEx[i].p21 = computeP21Grads(
        unit.xs[i], unit.ys[i],
        unit.pLocs[i],
        unit.pVals[i]
    );
    unit.gradPsEx[i].p22 = computeP22Grads(
        unit.xs[i], unit.ys[i],
        unit.pLocs[i],
        unit.pVals[i]
    );
  }

  // Gradient due to clamping
  for (size_t i(0), I(unit.size); i < I; i++) {
    if (unit.zs[i] <= 0.0 || unit.zs[i] >= 1.0) {
      unit.gradPsEx[i].p11 = 0.0;
      unit.gradPsEx[i].p12 = 0.0;
      unit.gradPsEx[i].p21 = 0.0;
      unit.gradPsEx[i].p22 = 0.0;
    }
  }

  // Gradient due to backprop
  for (size_t i(0), I(unit.size); i < I; i++) {
    if (unit.zs[i] <= 0.0 || unit.zs[i] >= 1.0) {
      unit.gradPsEx[i].p11 *= unit.gradZs[i];
      unit.gradPsEx[i].p12 *= unit.gradZs[i];
      unit.gradPsEx[i].p21 *= unit.gradZs[i];
      unit.gradPsEx[i].p22 *= unit.gradZs[i];
    }
  }
}

double computeXGrads(double x, double y, PLocs &pLocs, PVals &pVals) {
  return gradBlerpX(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

double computeYGrads(double x, double y, PLocs &pLocs, PVals &pVals) {
  return gradBlerpY(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

double computeP11Grads(double x, double y, PLocs &pLocs, PVals &pVals) {
  return gradBlerpP11(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

double computeP12Grads(double x, double y, PLocs &pLocs, PVals &pVals) {
  return gradBlerpP12(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

double computeP21Grads(double x, double y, PLocs &pLocs, PVals &pVals) {
  return gradBlerpP21(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

double computeP22Grads(double x, double y, PLocs &pLocs, PVals &pVals) {
  return gradBlerpP22(
      x, y,
      pLocs.x1, pLocs.x2,
      pLocs.y1, pLocs.y2,
      pVals.p11, pVals.p12,
      pVals.p21, pVals.p22
  );
}

/**************************** BATCH NORMALISATION *****************************/

void batchNorm(double *x, int n, double avg, double var) {
  if (var <= DBL_MIN) return;

  double  gamma = 0.25;
  double  beta  = 0.5;

  add(x, n, - avg);
  mul(x, n, gamma/sqrt(var));
  add(x, n, beta);
}

void batchNorm(Unit& unit) {
  // Compute stats
  double avg, var;
  moments(unit.zs, unit.size, avg, var);

  // Apply Bessel's correction
  double populationToSample = unit.size / ((double) unit.size - 1.0);
  var *= populationToSample;

  // TODO: Take moving average

  // Normalise activations (zs) and parameters (ps)
  batchNorm(unit.zs, unit.size,           avg, var);
  batchNorm(unit.ps, unit.res * unit.res, avg, var);
}

/******************************* REGULARISATION *******************************/

/*
 * Gradient of regularisation loss with respect to control points.
 */
void computeRegPGrads(Unit &unit) {
  double * const ps         = unit.ps;
  double * const gradPsReg  = unit.gradPsReg;
  double * const gradPsRegX = unit.gradPsRegX;

  for (int x(0), X(unit.res); x < X; x++) {
    for (int y(0), Y(unit.res); y < Y; y++) {
      if (x == 0) {
        gradPsRegX[P(X, x, y)] = 0.0              +
                                 ps[P(X, x+0, y)] +
                                 ps[P(X, x+1, y)];
      } else if (x == X - 1) {
        gradPsRegX[P(X, x, y)] = ps[P(X, x-1, y)] +
                                 ps[P(X, x+0, y)] +
                                 0.0;
      } else {
        gradPsRegX[P(X, x, y)] = ps[P(X, x-1, y)] +
                                 ps[P(X, x+0, y)] +
                                 ps[P(X, x+1, y)];
      }
    }
  }

  for (int x(0), X(unit.res); x < X; x++) {
    for (int y(0), Y(unit.res); y < Y; y++) {
      if (y == 0) {
        gradPsReg[P(X, x, y)] = 0.0                      +
                                gradPsRegX[P(X, x, y+0)] +
                                gradPsRegX[P(X, x, y+1)];
      } else if (y == Y - 1) {
        gradPsReg[P(X, x, y)] = gradPsRegX[P(X, x, y-1)] +
                                gradPsRegX[P(X, x, y+0)] +
                                0.0;
      } else {
        gradPsReg[P(X, x, y)] = gradPsRegX[P(X, x, y-1)] +
                                gradPsRegX[P(X, x, y+0)] +
                                gradPsRegX[P(X, x, y+1)];
      }
    }
  }

  for (int x(0), X(unit.res); x < X; x++) {
    for (int y(0), Y(unit.res); y < Y; y++) {
      bool isCorner = (x == 0 || x == X - 1) &&
                      (y == 0 || y == Y - 1);
      bool isEdge   = (x == 0 || x == X - 1  ||
                       y == 0 || y == Y - 1) &&
                      !isCorner;

      double c;
      if      (isCorner) c = 4.0;
      else if (isEdge)   c = 6.0;
      else               c = 9.0;

      gradPsReg[P(X, x, y)] = c * ps[P(X, x, y)] -
                           gradPsReg[P(X, x, y)];
    }
  }

  mul(gradPsReg, unit.res * unit.res , unit.reg);
}
