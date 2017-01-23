#ifndef UNIT_H
#define UNIT_H

#include <cstddef>
#include <cstdlib>
#include "more-math.h"
#include "blerp.h"

struct Bins { ptrdiff_t x; ptrdiff_t y; };
struct PIdxs { ptrdiff_t p11; ptrdiff_t p12; ptrdiff_t p21; ptrdiff_t p22; };
struct PLocs { double x1; double y1; double x2; double y2; };
struct PVals { double p11; double p12; double p21; double p22; };
struct PGrads { double p11; double p12; double p21; double p22; };

struct Unit {
  // Hyperparameters
  size_t size = 0;   // Batch size
  size_t res  = 0;   // "Resolution" of spline grid
  double  reg  = 0.0; // Regularisation strength
  bool   doBatchNorm = false;

  // Control Points
  double *ps = 0;

  // Inputs and outputs
  double *xs = 0;
  double *ys = 0;
  double *zs = 0;

  // Intermediate values
  Bins  *bins  = 0;
  PIdxs *pIdxs = 0; // TODO: Consider turning these into pointers
  PLocs *pLocs = 0;
  PVals *pVals = 0;

  // Spline gradients
  double *gradXs = 0;
  double *gradYs = 0;
  double *gradZs = 0; // Back-propagated gradient
  double *gradPs = 0;

  // Regularisation gradients
  PGrads *gradPsEx   = 0;
  double  *gradPsReg  = 0;
  double  *gradPsRegX = 0;
};

ptrdiff_t P(size_t res, ptrdiff_t i, ptrdiff_t j);
Unit makeUnit(size_t size, size_t res, double reg=0.0, bool doBatchNorm=false);
void freeUnit(Unit &unit);
void initUnit(Unit &unit);
void setSize(Unit &unit, size_t size);
void setRes(Unit &unit, size_t res);
void setReg(Unit &unit, double reg);
void setBatchNorm(Unit &unit, bool doBatchNorm);
void setGradZs(Unit &unit, double *gradZs);
void setXs(Unit &unit, double *xs);
void setYs(Unit &unit, double *ys);
void setInputs(Unit &unit, double *xs, double *ys);
void eval(Unit &unit, double *xs, double *ys, bool doBatchNorm=false);
void eval(Unit &unit);
void computeBins(Unit &unit);
ptrdiff_t bin(double x, size_t res);
void computePIdxs(Unit &unit);
void computePIdxs(Bins &bins, PIdxs &pIdxs, size_t res);
void computePLocs(Unit &unit);
void computePLocs(double x, double y, PLocs &pLocs, size_t res);
void computePVals(Unit &unit);
void computePVals(double *ps, PIdxs &pIdxs, PVals &pVals);
void computeZs(Unit &unit);
double blerp(double x, double y, PLocs &pLocs, PVals &pVals);

/**************************** GRADIENT COMPUTATION ****************************/

void computeGrads(Unit &unit, double *gradZs);
void computeGrads(Unit &unit);
void computeXGrads(Unit &unit);
void computeYGrads(Unit &unit);
void computePGrads(Unit &unit);
void computeExPGrads(Unit &unit);
double computeXGrads(double x, double y, PLocs &pLocs, PVals &pVals);
double computeYGrads(double x, double y, PLocs &pLocs, PVals &pVals);
double computeP11Grads(double x, double y, PLocs &pLocs, PVals &pVals);
double computeP12Grads(double x, double y, PLocs &pLocs, PVals &pVals);
double computeP21Grads(double x, double y, PLocs &pLocs, PVals &pVals);
double computeP22Grads(double x, double y, PLocs &pLocs, PVals &pVals);

/**************************** BATCH NORMALISATION *****************************/

void batchNorm(double *x, int n, double avg, double var);
void batchNorm(Unit& unit);

/******************************* REGULARISATION *******************************/

void computeRegPGrads(Unit &unit);

#endif
