#include "mnist.h"
#include "more-math.h"
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>

struct ExampleGrads { float xlyl; float xuyl; float xlyu; float xuyu; };
struct WGrads       { float xlyl; float xuyl; float xlyu; float xuyu; };
struct CPtLocs      { int   xbl;  int   xbu;  int   ybl;  int   ybu;  };
struct CPtVals      { float xlyl; float xuyl; float xlyu; float xuyu; };
struct CPtDists     { float xdl;  float xdu;  float ydl;  float ydu;  };
struct Moments      { float avg;  float var;                          };

struct Net {
  int   dim; // Dimensionality of input
  int   res;
  float reg;
  int   depth;
  int   numUnits;
  int   unitSize;

  float    *params;
  CPtLocs  *cPtLocs;
  CPtVals  *cPtVals;
  CPtDists *cPtDists;
  float    *acts;

  //Optimisation variables
  float **inputs;
  float  *targets;

  float **batchInputs;
  float  *batchOutputs;
  float  *batchTargets;

  int   numExamples;
  float rate;
  float momentum;
  int   batchSize;
  int   batchIndex;

  float        *step;
  float         batchMse;
  Moments      *stats;
  float        *dError;
  float        *batchGrads;
  float        *regGrads;
  float        *xRegGrads;
  ExampleGrads *exampleGrads;
  float        *xGrads;
  float        *yGrads;
  WGrads       *wGrads;
  float        *backGrads;
};

std::default_random_engine generator;

// TODO: Improve numerical stability and re-check gradients numerically
/***************************** COUNTING FUNCTIONS *****************************/

int lenLayer(Net& net, int l) {
  return pow(2, net.depth - l - 1);
}

// Number of units above the layer indexed by `l`
int numUnitsAbove(Net& net, int l) {
  return net.dim - pow(2, net.depth - l);
}

// Number of units before the unit at layer `l`, index `i`
int numUnitsAbove(Net& net, int l, int i) {
  return numUnitsAbove(net, l) + i;
}

/***************************** INDEXING FUNCTIONS *****************************/

int I(Net& net, int l, int i) {
  return numUnitsAbove(net, l, i);
}

int I(Net& net, int l) {
  return I(net, l, 0);
}

// Indexes the elements (pixels) in a unit
int I_unit(Net& net, int x, int y) {
  return net.res * x + y;
}

int I_params(Net& net, int l, int i, int x, int y) {
  return (net.unitSize) * I(net, l, i) + I_unit(net, x, y);
}

int I_params(Net& net, int l, int i) {
  return I_params(net, l, i, 0, 0);
}

int I_params(Net& net, int l) {
  return I_params(net, l, 0);
}

/********************************** GETTERS ***********************************/

float* getUnit(Net& net, int l, int i) {
  return &net.params[I_params(net, l, i)];
}

float* getUnit(Net& net, int l) {
  return getUnit(net, l, 0);
}

float getParam(Net& net, int l, int i, int x, int y) {
  return net.params[I_params(net, l, i, x, y)];
}

Moments* getStats(Net& net, int l) {
  return &net.stats[I(net, l, 0)];
}

float* getActs(Net& net, int l, int i) {
  return &net.acts[net.batchSize*I(net, l, i)];
}

float* getActs(Net& net, int l) {
  return getActs(net, l, 0);
}

CPtLocs* getCPtLocs(Net& net, int l, int i) {
  return &net.cPtLocs[net.batchSize*I(net, l, i)];
}

CPtLocs* getCPtLocs(Net& net, int l) {
  return getCPtLocs(net, l, 0);
}

CPtVals* getCPtVals(Net& net, int l, int i) {
  return &net.cPtVals[net.batchSize*I(net, l, i)];
}

CPtVals* getCPtVals(Net& net, int l) {
  return getCPtVals(net, l, 0);
}

CPtDists* getCPtDists(Net& net, int l, int i) {
  return &net.cPtDists[net.batchSize*I(net, l, i)];
}

CPtDists* getCPtDists(Net& net, int l) {
  return getCPtDists(net, l, 0);
}

float* getXGrads(Net& net, int l, int i) {
  return &net.xGrads[net.batchSize*I(net, l, i)];
}

float* getXGrads(Net& net, int l) {
  return &net.xGrads[net.batchSize*I(net, l, 0)];
}

float* getYGrads(Net& net, int l, int i) {
  return &net.yGrads[net.batchSize*I(net, l, i)];
}

float* getYGrads(Net& net, int l) {
  return &net.yGrads[net.batchSize*I(net, l, 0)];
}

float* getBackGrads(Net& net, int l, int i) {
  return &net.backGrads[net.batchSize*I(net, l, i)];
}

float* getBackGrads(Net& net, int l) {
  return &net.backGrads[net.batchSize*I(net, l, 0)];
}

/********************************** PRINTING **********************************/

void printParams(std::ofstream& file, Net& net, int l, int i, int x) {
  file << "      [";
  for (int y = 0; y < net.res; y++) {
    int precision = std::numeric_limits<float>::max_digits10;
    float param = getParam(net, l, i, x, y);

    file << std::setprecision(precision) << param << ", ";
  }
  file << "]";
}

void printParams(std::ofstream& file, Net& net, int l, int i) {
  file << "    [" << std::endl;
  for (int x = 0; x < net.res; x++) {
    printParams(file, net, l, i, x);
    file << "," << std::endl;
  }
  file << "    ]";
}

void printParams(std::ofstream& file, Net& net, int l) {
  file << "  [" << std::endl;
  for (int i = 0; i < lenLayer(net, l); i++) {
    printParams(file, net, l, i);
    file << "," << std::endl;
  }
  file << "  ]";
}

void printParams(std::string filename, Net& net) {
  std::ofstream file;
  file.open(filename);

  file << "net = ";
  file << "[" << std::endl;
  for (int i = 0; i < net.depth; i++) {
    printParams(file, net, i);
    file << "," << std::endl;
  }
  file << "]";

  file.close();
}

void printParams(int i, Net& net) {
  // Convert index `i` into file filename
  char filename[100];
  snprintf(filename, sizeof(filename), "plots/net-%09d.net", i);
  std::string filenameAsStdStr = filename;

  printParams(filenameAsStdStr, net);
}

/*************************** NETWORK INITIALISATION ***************************/

void allocNet(Net& net) {
  // Model
  net.params = new float[net.numUnits*net.unitSize];

  net.cPtLocs  = new CPtLocs [net.batchSize*net.numUnits];
  net.cPtVals  = new CPtVals [net.batchSize*net.numUnits];
  net.cPtDists = new CPtDists[net.batchSize*net.numUnits];
  net.acts     = new float   [net.batchSize*net.numUnits];

  // Optimiser
  net.stats        = new Moments     [net.numUnits];
  net.dError       = new float       [net.batchSize];
  net.batchGrads   = new float       [              net.numUnits*net.unitSize];
  net.exampleGrads = new ExampleGrads[net.batchSize*net.numUnits];
  net.regGrads     = new float       [              net.numUnits*net.unitSize];
  net.xRegGrads    = new float       [              net.numUnits*net.unitSize];
  net.xGrads       = new float       [net.batchSize*net.numUnits];
  net.yGrads       = new float       [net.batchSize*net.numUnits];
  net.wGrads       = new WGrads      [net.batchSize*net.numUnits];
  net.backGrads    = new float       [net.batchSize*net.numUnits];
  net.step         = new float       [              net.numUnits*net.unitSize];
}

void initUnit(Net& net, int l, int i) {
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  float *unit = getUnit(net, l, i);
  for (int x = 0; x < net.res; x++)
    for (int y = 0; y < net.res; y++) {
      float fx = x / (net.res - 1.0);
      float fy = y / (net.res - 1.0);
      if (fx > 0.5 != fy > 0.5) unit[I_unit(net, x, y)] = 0.0;
      else                      unit[I_unit(net, x, y)] = 1.0;

      unit[I_unit(net, x, y)] = distribution(generator);
      //unit[I_unit(net, x, y)] = (fx + fy) / 2.0;
    }
}

void initNet(Net& net) {
  for (int i = 0; i < net.depth; i++)
    for (int j = 0; j < lenLayer(net, i); j++)
      initUnit(net, i, j);
}

Net makeNet(
    int dim, int res, float reg,
    float **inputs, float *targets, int numExamples,
    float rate, float momentum, int batchSize
) {
  Net net;

  // Model variables
  net.depth    = ceil(log2(dim));
  net.dim      = pow(2, net.depth);
  net.res      = res;
  net.reg      = reg;
  net.numUnits = net.dim - 1;
  net.unitSize = res * res;

  // Optimisation variables
  net.inputs      = inputs;
  net.targets     = targets;
  net.numExamples = numExamples;
  net.rate        = rate;
  net.momentum    = momentum;
  net.batchSize   = batchSize;
  net.batchIndex  = numExamples/batchSize; // Past last batch. Triggers shuffle

  // Allocation
  allocNet(net);
  initNet (net);

  net.batchOutputs = getActs(net, net.depth - 1);

  return net;
}

/******************** INPUT EVALUATION (FEED-FORWARD PASS) ********************/

void binPair(
    Net& net, float x,
    int& xbl, int& xbu
) {
  xbl = (int) (x * (net.res - 1));
  xbl = clamp(xbl, 0, net.res - 2);

  xbu = xbl + 1;
}

void binPairs(
    Net& net, float x, float y,
    int& xbl, int& xbu, int& ybl, int& ybu
) {
  binPair(net, x, xbl, xbu);
  binPair(net, y, ybl, ybu);
}

void distPair(
    Net& net, float x,     // Inputs
    int xbl, int xbu,
    float &xdl, float &xdu // Outputs
) {
  float xsl = xbl / (net.res - 1.0); // Snapped upper
  float xsu = xbu / (net.res - 1.0); // Snapped lower

  xdl = x - xsl;
  xdu = xsu - x;
}

void distPairs(
    Net& net, float x, float y,                    // Inputs
    int xbl, int xbu, int ybl, int ybu,
    float &xdl, float &xdu, float &ydl, float &ydu // Outputs
) {
  distPair(net, x, xbl, xbu, xdl, xdu);
  distPair(net, y, ybl, ybu, ydl, ydu);
}

void computeCPtLocsInput(Net& net) {
  // Figure out where the previous layer of `acts` and `cPtLocs` is in memory
  CPtLocs  *cPtLocs = getCPtLocs(net, 0);
  float   **acts    = net.batchInputs;

  // Compute act locs
  for (int i(0), I(lenLayer(net, 0)); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      binPairs(
          net,
          acts[j][2*i+0], acts[j][2*i+1],
          cPtLocs[i*J+j].xbl, cPtLocs[i*J+j].xbu, cPtLocs[i*J+j].ybl, cPtLocs[i*J+j].ybu
      );
    }
  }
}

void computeCPtLocsHidden(Net& net, int l) {
  // Figure out where the previous layer of `acts` and `cPtLocs` is in memory
  CPtLocs *cPtLocs = getCPtLocs(net, l);
  float   *acts    = getActs   (net, l-1);

  // Compute act locs
  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      binPairs(
          net,
          acts[(2*i+0)*J + j], acts[(2*i+1)*J + j],
          cPtLocs[i*J+j].xbl, cPtLocs[i*J+j].xbu, cPtLocs[i*J+j].ybl, cPtLocs[i*J+j].ybu
      );
    }
  }
}

void computeCPtLocs(Net& net, int l) {
  if (l == 0) computeCPtLocsInput (net);
  else        computeCPtLocsHidden(net, l);
}

void computeCPtDistsInput(Net& net) {
  CPtLocs   *cPtLocs  = getCPtLocs (net, 0);
  CPtDists  *cPtDists = getCPtDists(net, 0);
  float    **acts     = net.batchInputs;

  // Compute `cPtDists`
  for (int i(0), I(lenLayer(net, 0)); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      distPairs(
          net,
          acts[j][2*i+0], acts[j][2*i+1],
          cPtLocs [i*J+j].xbl, cPtLocs [i*J+j].xbu, cPtLocs [i*J+j].ybl, cPtLocs [i*J+j].ybu,
          cPtDists[i*J+j].xdl, cPtDists[i*J+j].xdu, cPtDists[i*J+j].ydl, cPtDists[i*J+j].ydu
      );
    }
  }
}

void computeCPtDistsHidden(Net& net, int l) {
  CPtLocs  *cPtLocs  = getCPtLocs (net, l);
  CPtDists *cPtDists = getCPtDists(net, l);
  float    *acts     = getActs    (net, l-1);

  // Compute `cPtDists`
  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      distPairs(
          net,
          acts[(2*i+0)*J + j], acts[(2*i+1)*J + j],
          cPtLocs [i*J+j].xbl, cPtLocs [i*J+j].xbu, cPtLocs [i*J+j].ybl, cPtLocs [i*J+j].ybu,
          cPtDists[i*J+j].xdl, cPtDists[i*J+j].xdu, cPtDists[i*J+j].ydl, cPtDists[i*J+j].ydu
      );
    }
  }
}

void computeCPtDists(Net& net, int l) {
  if (l == 0) computeCPtDistsInput (net);
  else        computeCPtDistsHidden(net, l);
}

void computeCPtVals(Net& net, int l) {
  CPtLocs *cPtLocs  = getCPtLocs(net, l);
  CPtVals *cPtVals  = getCPtVals(net, l);
  float   *unit     = getUnit   (net, l);

  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      int xbl = cPtLocs[i*J+j].xbl;
      int xbu = cPtLocs[i*J+j].xbu;
      int ybl = cPtLocs[i*J+j].ybl;
      int ybu = cPtLocs[i*J+j].ybu;

      cPtVals[i*J+j].xlyl = unit[I_unit(net, xbl, ybl)];
      cPtVals[i*J+j].xuyl = unit[I_unit(net, xbu, ybl)];
      cPtVals[i*J+j].xlyu = unit[I_unit(net, xbl, ybu)];
      cPtVals[i*J+j].xuyu = unit[I_unit(net, xbu, ybu)];
    }

    unit += net.unitSize;
  }
}

void computeActs(Net& net, int l) {
  CPtVals  *cPtVals  = getCPtVals (net, l);
  CPtDists *cPtDists = getCPtDists(net, l);
  float    *acts     = getActs    (net, l);
  float    norm      = (net.res - 1) * (net.res - 1);

  for (int i(0), I(net.batchSize * lenLayer(net, l)); i < I; i++) {
    acts[i] = norm * (
        cPtVals[i].xlyl * cPtDists[i].xdu * cPtDists[i].ydu +
        cPtVals[i].xuyl * cPtDists[i].xdl * cPtDists[i].ydu +
        cPtVals[i].xlyu * cPtDists[i].xdu * cPtDists[i].ydl +
        cPtVals[i].xuyu * cPtDists[i].xdl * cPtDists[i].ydl
    );
  }
  // TODO: Clamp is in the wrong place
  for (int i(0), I(net.batchSize * lenLayer(net, l)); i < I; i++)
    acts[i] = clamp(acts[i], 0.0, 1.0);
}

void batchNorm(float *x, int n, float avg, float var) {
  if (var <= FLT_MIN) return;

  float  gamma = 0.25;
  float  beta  = 0.5;

  add(x, n, - avg);
  mul(x, n, gamma/sqrt(var));
  add(x, n, beta);
}

void batchNorm(Net& net, int l) {
  if (net.batchSize <= 1) return;

  float *unit;
  float *acts;
  Moments *stats = getStats(net, l);
  int       i;
  const int L = lenLayer(net, l);

  // Compute layer stats
  for (i = 0, acts = getActs(net, l); i < L; i++, acts += net.batchSize)
    moments(acts, net.batchSize, stats[i].avg, stats[i].var);

  // Apply Bessel's correction
  float populationToSample = net.batchSize / ((float) net.batchSize - 1.0);
  for (int i = 0; i < L; i++)
    stats[i].var *= populationToSample;

  // TODO: Take moving average

  // Normalise activations
  for (i = 0, acts = getActs(net, l); i < L; i++, acts += net.batchSize)
    batchNorm(acts, net.batchSize, stats[i].avg, stats[i].var);

  // Normalise units
  for (i = 0, unit = getUnit(net, l); i < L; i++, unit += net.unitSize)
    batchNorm(unit, net.unitSize, stats[i].avg, stats[i].var);
}

void forward(Net& net, bool doBatchNorm) {
  for (int i(0), I(net.depth); i < I; i++) {
    computeCPtLocs (net, i);
    computeCPtDists(net, i);
    computeCPtVals (net, i);
    computeActs    (net, i);
    if (doBatchNorm && i < I - 1)
    batchNorm      (net, i);
  }
}

void forward(Net& net, float** batchInputs, bool doBatchNorm=false) {
  net.batchInputs = batchInputs;
  forward(net, doBatchNorm);
}

void forward(Net& net, float* input) {
  // Start using a batch size of 1
  int oldBatchSize = net.batchSize;
  net.batchSize = 1;

  // Forward-prop and copy the output to the usual place (`net.batchOutputs`)
  forward(net, &input);
  float *output = getActs(net, net.depth - 1);
  net.batchOutputs[0] = output[0];

  // Restore batch size
  net.batchSize = oldBatchSize;
}

/**************************** GRADIENT COMPUTATION ****************************/

void computeXGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  CPtVals  *cPtVals  = net.cPtVals;
  float     c        = (net.res - 1.0) * (net.res - 1.0);

  for (int i(0), I(net.batchSize * net.numUnits); i < I; i++) {
    net.xGrads[i] = c * (
        cPtDists[i].ydu * (cPtVals[i].xuyl - cPtVals[i].xlyl) +
        cPtDists[i].ydl * (cPtVals[i].xuyu - cPtVals[i].xlyu)
    );
  }

  for (int i(0), I(net.batchSize * net.numUnits); i < I; i++) {
    if (net.acts[i] > 1.0) net.xGrads[i] = 0.0;
    if (net.acts[i] < 0.0) net.xGrads[i] = 0.0;
  }
}

void computeYGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  CPtVals  *cPtVals  = net.cPtVals;
  float     c        = (net.res - 1.0) * (net.res - 1.0);

  for (int i(0), I(net.batchSize * net.numUnits); i < I; i++) {
    net.yGrads[i] = c * (
        cPtDists[i].xdu * (cPtVals[i].xlyu - cPtVals[i].xlyl) +
        cPtDists[i].xdl * (cPtVals[i].xuyu - cPtVals[i].xuyl)
    );
  }

  for (int i(0), I(net.batchSize * net.numUnits); i < I; i++) {
    if (net.acts[i] > 1.0) net.yGrads[i] = 0.0;
    if (net.acts[i] < 0.0) net.yGrads[i] = 0.0;
  }
}

void computeWGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  WGrads   *wGrads   = net.wGrads;
  float    norm      = (net.res - 1) * (net.res - 1);

  for (int i(0), I(net.batchSize * net.numUnits); i < I; i++) {
    wGrads[i].xlyl = norm * cPtDists[i].xdu * cPtDists[i].ydu;
    wGrads[i].xlyu = norm * cPtDists[i].xdu * cPtDists[i].ydl;
    wGrads[i].xuyl = norm * cPtDists[i].xdl * cPtDists[i].ydu;
    wGrads[i].xuyu = norm * cPtDists[i].xdl * cPtDists[i].ydl;
  }
}

float dError(float output, float target) {
  return output - target;
}

void computeBackGrads(Net& net) {
  {
    float *backGrads = getBackGrads(net, net.depth - 1);
    for (int i = net.batchSize - 1; i >= 0; i--)
      backGrads[i] = 1.0;
  }

  for (int i = net.depth - 1; i >= 1; i--) {
    int L = lenLayer(net, i);

    float *xGrads        = getXGrads   (net, i);
    float *yGrads        = getYGrads   (net, i);
    float *prevBackGrads = getBackGrads(net, i-1);
    float *nextBackGrads = getBackGrads(net, i);

    for (int j = L-1; j >= 0; j--) {
      for (int k(net.batchSize - 1), K(net.batchSize); k >= 0; k--) {
        prevBackGrads[(2*j+0)*K + k] = /*sgn*/(nextBackGrads[j*K + k]) * xGrads[j*K + k];
        prevBackGrads[(2*j+1)*K + k] = /*sgn*/(nextBackGrads[j*K + k]) * yGrads[j*K + k];
      }
    }
  }

  for (int i(0), I(net.batchSize); i < I; i++)
    net.dError[i] = dError(net.batchOutputs[i], net.batchTargets[i]);

  for (int i(0), I(net.numUnits); i < I; i++)
    for (int j(0), J(net.batchSize); j < J; j++)
      net.backGrads[i*J+j] *= net.dError[j];
}

void computeExampleGrads(Net& net) {
  ExampleGrads *exampleGrads = net.exampleGrads;
  float        *backGrads    = net.backGrads;
  WGrads       *wGrads       = net.wGrads;

  for (int i(0), I(net.batchSize * net.numUnits); i < I; i++) {
    exampleGrads[i].xlyl = backGrads[i] * wGrads[i].xlyl;
    exampleGrads[i].xlyu = backGrads[i] * wGrads[i].xlyu;
    exampleGrads[i].xuyl = backGrads[i] * wGrads[i].xuyl;
    exampleGrads[i].xuyu = backGrads[i] * wGrads[i].xuyu;
  }
}

void backward(Net& net) {
  computeXGrads      (net);
  computeYGrads      (net);
  computeWGrads      (net);
  computeBackGrads   (net);
  computeExampleGrads(net);
}

void backward(Net& net, float* targets) {
  net.batchTargets = targets;
  backward(net);
}

// TODO: Profile function
void computeRegGrads(Net& net) {
  float *unit;
  float *xRegGrads;
  float *regGrads;

  for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    net.regGrads[i] = 0.0;

  // Compute regularisation along x direction
  unit      = net.params;
  xRegGrads = net.xRegGrads;
  for (int i(0), I(net.numUnits); i < I; i++) {
    for (int x(0), X(net.res); x < X; x++) {
      for (int y(0), Y(net.res); y < Y; y++) {
        if (x == 0) {
          xRegGrads[I_unit(net, x, y)] = unit[I_unit(net, x+0, y)] +
                                         unit[I_unit(net, x+1, y)];
        } else if (x == X - 1) {
          xRegGrads[I_unit(net, x, y)] = unit[I_unit(net, x-1, y)] +
                                         unit[I_unit(net, x+0, y)];
        } else {
          xRegGrads[I_unit(net, x, y)] = unit[I_unit(net, x-1, y)] +
                                         unit[I_unit(net, x+0, y)] +
                                         unit[I_unit(net, x+1, y)];
        }
      }
    }
    unit      += net.unitSize;
    xRegGrads += net.unitSize;
  }

  // Compute regularisation along y direction
  xRegGrads = net.xRegGrads;
  regGrads  = net.regGrads;
  for (int i(0), I(net.numUnits); i < I; i++) {
    for (int x(0), X(net.res); x < X; x++) {
      for (int y(0), Y(net.res); y < Y; y++) {
        if (y == 0) {
          regGrads[I_unit(net, x, y)] = xRegGrads[I_unit(net, x, y+0)] +
                                        xRegGrads[I_unit(net, x, y+1)];
        } else if (y == Y - 1) {
          regGrads[I_unit(net, x, y)] = xRegGrads[I_unit(net, x, y-1)] +
                                        xRegGrads[I_unit(net, x, y+0)];
        } else {
          regGrads[I_unit(net, x, y)] = xRegGrads[I_unit(net, x, y-1)] +
                                        xRegGrads[I_unit(net, x, y+0)] +
                                        xRegGrads[I_unit(net, x, y+1)];
        }
      }
    }
    xRegGrads += net.unitSize;
    regGrads  += net.unitSize;
  }

  // Compute complete regularisation
  regGrads = net.regGrads;
  unit     = net.params;
  for (int i(0), I(net.numUnits); i < I; i++) {
    for (int x(0), X(net.res); x < X; x++) {
      for (int y(0), Y(net.res); y < Y; y++) {
        bool isCorner = (x == 0 || x == X - 1) &&
                        (y == 0 || y == Y - 1);
        bool isEdge   = (x == 0 || x == X - 1  ||
                         y == 0 || y == Y - 1) &&
                        !isCorner;

        float c;
        if      (isCorner) c = 4.0;
        else if (isEdge)   c = 6.0;
        else               c = 9.0;

        regGrads[I_unit(net, x, y)] = c * unit[I_unit(net, x, y)] -
                                      regGrads[I_unit(net, x, y)];
      }
    }
    unit     += net.unitSize;
    regGrads += net.unitSize;
  }

  // Scale gradients by regularisation hyper-parameter
  mul(net.regGrads, net.numUnits * net.unitSize, net.reg);
}

void addExampleGrads(Net& net) {
  int           batchSize    = net.batchSize;
  float        *batchGrads   = net.batchGrads;
  ExampleGrads *exampleGrads = net.exampleGrads;
  CPtLocs      *cPtLocs      = net.cPtLocs;

  for (int i(0), I(net.numUnits); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      int xbl = cPtLocs[i*J+j].xbl;
      int xbu = cPtLocs[i*J+j].xbu;
      int ybl = cPtLocs[i*J+j].ybl;
      int ybu = cPtLocs[i*J+j].ybu;

      batchGrads[I_unit(net, xbl, ybl)] += exampleGrads[i*J+j].xlyl / batchSize;
      batchGrads[I_unit(net, xbu, ybl)] += exampleGrads[i*J+j].xuyl / batchSize;
      batchGrads[I_unit(net, xbl, ybu)] += exampleGrads[i*J+j].xlyu / batchSize;
      batchGrads[I_unit(net, xbu, ybu)] += exampleGrads[i*J+j].xuyu / batchSize;
    }

    batchGrads += net.unitSize;
  }
}

void addExampleError(Net& net) {
  for (int i(0), I(net.batchSize); i < I; i++) {
    float dErr = net.batchOutputs[i] - net.batchTargets[i];
    net.batchMse += dErr * dErr / I / 2.0;
  }
}

void shuffleExamples(Net& net) {
  for (int i = 0, n(net.numExamples - 1); i < n; i++) {
    std::uniform_int_distribution<int> distribution(i, n - 1);
    int j = distribution(generator);

    std::swap(net.inputs [i], net.inputs [j]);
    std::swap(net.targets[i], net.targets[j]);
  }
}

void initBatchVars(Net& net) {
  // Shuffle examples
  bool epochDone = net.batchSize * (net.batchIndex + 1) > net.numExamples;
  if (epochDone) {
    //shuffleExamples(net); TODO: Uncomment
    net.batchIndex = 0;
  }

  // Zero-out batchGrads
  for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    net.batchGrads[i] = 0.0;

  // Zero-out batchMse
  net.batchMse = 0.0;
}

void computeBatchGrads(Net& net) {
  initBatchVars(net);

  // Get loop bounds and increment batchIndex for the next call
  int i = net.batchSize * net.batchIndex;
  net.batchIndex++;

  forward (net, &net.inputs[i], true);
  backward(net, &net.targets[i]);
  addExampleGrads(net);
  addExampleError(net);
}

float obj(Net& net, float **inputs, float *targets) {
  float result = 0.0;

  forward(net, inputs);

  for (int i(0), I(net.batchSize); i < I; i++) {
    float dErr = net.batchOutputs[i] - targets[i];
    result += dErr * dErr / I / 2.0;
  }

  return result;
}

void computeBatchGradsNumerically(Net& net) {
  initBatchVars(net);

  // Get loop bounds and increment batchIndex for the next call
  int i = net.batchSize * net.batchIndex;
  net.batchIndex++;

  // Performs batch normalisation
  forward(net, &net.inputs[i], true);

  for (int j(0), J(net.unitSize * net.numUnits); j < J; j++) {
    float h = 1e-5;
    float objSub;
    float objAdd;

    float prev = net.params[j];
    net.params[j] = prev + h; objAdd = obj(net, &net.inputs[i], &net.targets[i]);
    net.params[j] = prev - h; objSub = obj(net, &net.inputs[i], &net.targets[i]);
    net.params[j] = prev;

    net.batchGrads[j] = (objAdd - objSub) / (2.0 * h);
  }
}

void checkGradients(Net& net) {
  computeBatchGrads(net);

  // Copy `net.batchGrads`
  int sizeOfBatchGrads = net.numUnits*net.unitSize;
  float *batchGrads = new float[sizeOfBatchGrads];
  for (int i = 0; i < sizeOfBatchGrads; i++)
    batchGrads[i] = net.batchGrads[i];

  // Rewind batch index
  net.batchIndex--;

  computeBatchGradsNumerically(net);

  // Compare gradients
  // TODO: Display unit index
  float threshold = 1e-2;
  for (int i = 0; i < sizeOfBatchGrads; i++) {
    float diff = net.batchGrads[i] - batchGrads[i];
    float adiff = fabs(diff);
    if (adiff > threshold) {
      std::cout
        << "Grad "
        << i
        << "<"
        << sizeOfBatchGrads
        << " in batch "
        << net.batchIndex - 1
        << " differs from numerical gradient by "
        << adiff
        << ". Wanted: "
        << net.batchGrads[i]
        << ". Got: "
        <<     batchGrads[i]
        << "."
        << std::endl;
    }
  }

  delete[] batchGrads;
}

void sgd(Net& net) {
  computeRegGrads(net);
  computeBatchGrads(net);

  // Compute step
  for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    net.step[i] = net.momentum * net.step[i] -
                  net.rate * (net.batchGrads[i] + net.regGrads[i]);

  // Update params
  for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    net.params[i] += net.step[i];
}

/******************************** BENCHMARKING ********************************/

float classificationError(Net& net, float* input, float target) {
  forward(net, input);

  if (roundf(net.batchOutputs[0]) == target) return 0.0;
  return 1.0;
}

float classificationError(
    Net& net,
    float** inputs,
    float* targets,
    int numExamples,
    int sampleSize=5000
) {
  sampleSize = min(sampleSize, numExamples);

  float sum = 0.0;
  for (int i = 0; i < sampleSize; i++) {
    std::uniform_int_distribution<int> distribution(0, sampleSize-1);
    int j = distribution(generator);

    sum += classificationError(net, inputs[j], targets[j]) / sampleSize;
  }
  return sum;
}

/******************************* SYNTHETIC DATA *******************************/

int numOnes(float *input, int dim) {
  int n = 0;
  for (int i = 0; i < dim; i++)
    if (input[i] >= 0.5)
      n++;
  return n;
}

float trueClassifier(float *input, int dim) {
  if (input[0] == max(input, dim)) return 1.0;
  //if (numOnes(input, dim) % 2 == 0) return 1.0;

  return 0.0;
}

void makeData(float **&inputs, float *&targets, int dim, int numExamples) {
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Allocate inputs
  inputs = new float*[numExamples];
  for (int i = 0; i < numExamples; i++)
    inputs[i] = new float[dim];

  // Allocate targets
  targets = new float[numExamples];

  // Init inputs
  for (int i = 0; i < numExamples; i++)
    for(int j = 0; j < dim; j++)
      inputs[i][j] = distribution(generator);

  // Init targets
  for (int i = 0; i < numExamples; i++)
    targets[i] = trueClassifier(inputs[i], dim);
}

/************************************ MAIN ************************************/

int main() {
  // MODEL VARS
  int   dim = 4;
  int   res = 10;
  float reg = 0.1;

  // OPTIMISER VARS
  float rate      = 0.1;
  float momentum  = 0.0;
  int   batchSize = 1;

  // LOAD SYNTHETIC DATA SET
  int     numExamplesTrn = 100000;
  float** inputsTrn;
  float*  targetsTrn;

  makeData(inputsTrn, targetsTrn, dim, numExamplesTrn);

  // Make model
  Net net = makeNet(
      dim, res, reg,
      inputsTrn, targetsTrn, numExamplesTrn,
      rate, momentum, batchSize
  );

  for (int i = 0; i < 1000; i++)
  checkGradients(net);
  return 0;

  /* TEST */
  //printParams(0, net);
  //forward(net, inputsTrn);

  //for (int i = 0; i < batchSize; i++) {
    //std::cout
      //<< inputsTrn[i][0] << " "
      //<< inputsTrn[i][1] << " "
      //<< inputsTrn[i][2] << " "
      //<< inputsTrn[i][3] << " "
      //<< std::endl;
    //std::cout << net.batchOutputs[i] << std::endl;
    //std::cout << std::endl;
  //}

  //std::cout << net.batchMse << std::endl;

  //float e;
  //e = classificationError(net, inputsTrn, targetsTrn, batchSize, batchSize);
  //e *= 100;
  //std::cout << "Train error (%): " << e << std::endl;

  //std::cout << std::endl;

  //return 0;
  /* TEST */

  // OPTIMISE
  for (int i = 0; i < 10000; i++) {
    for (int j = 0; j < 100; j++)
      sgd(net);
    net.reg *= 0.99;

    std::cout << net.batchMse << std::endl;

    float e;
    e = classificationError(net, inputsTrn, targetsTrn, numExamplesTrn, numExamplesTrn);
    e *= 100;
    std::cout << "Train error (%): " << e << std::endl;

    std::cout << std::endl;

    printParams(i, net);
  }
}
