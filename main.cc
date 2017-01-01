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

struct Net {
  int   dim; // Dimensionality of input
  int   res;
  float reg;
  int   depth;
  int   numUnits;
  int   unitSize;

  float *input;
  float *output;

  float    *params;
  CPtLocs  *cPtLocs;
  CPtVals  *cPtVals;
  CPtDists *cPtDists;
  float    *acts;

  //Optimisation variables
  float **inputs;
  float  *outputs;

  int   numExamples;
  float rate;
  float momentum;
  int   batchSize;
  int   batchIndex;

  float        *step;
  float         batchMse;
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

float* getActs(Net& net, int l, int i) {
  if (l < 0) return &net.input[I(net, 0, i)];
  else       return &net.acts [I(net, l, i)];
}

float* getActs(Net& net, int l) {
  return getActs(net, l, 0);
}

CPtLocs* getCPtLocs(Net& net, int l, int i) {
  return &net.cPtLocs[I(net, l, i)];
}

CPtLocs* getCPtLocs(Net& net, int l) {
  return getCPtLocs(net, l, 0);
}

CPtVals* getCPtVals(Net& net, int l, int i) {
  return &net.cPtVals[I(net, l, i)];
}

CPtVals* getCPtVals(Net& net, int l) {
  return getCPtVals(net, l, 0);
}

CPtDists* getCPtDists(Net& net, int l, int i) {
  return &net.cPtDists[I(net, l, i)];
}

CPtDists* getCPtDists(Net& net, int l) {
  return getCPtDists(net, l, 0);
}

float* getXGrads(Net& net, int l, int i) {
  return &net.xGrads[I(net, l, i)];
}

float* getXGrads(Net& net, int l) {
  return &net.xGrads[I(net, l, 0)];
}

float* getYGrads(Net& net, int l, int i) {
  return &net.yGrads[I(net, l, i)];
}

float* getYGrads(Net& net, int l) {
  return &net.yGrads[I(net, l, 0)];
}

float* getBackGrads(Net& net, int l, int i) {
  return &net.backGrads[I(net, l, i)];
}

float* getBackGrads(Net& net, int l) {
  return &net.backGrads[I(net, l, 0)];
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
  net.batchGrads   = new float       [net.batchSize*net.numUnits*net.unitSize];
  net.exampleGrads = new ExampleGrads[net.batchSize*net.numUnits];
  net.regGrads     = new float       [              net.numUnits*net.unitSize];
  net.xRegGrads    = new float       [              net.numUnits*net.unitSize];
  net.xGrads       = new float       [net.batchSize*net.numUnits];
  net.yGrads       = new float       [net.batchSize*net.numUnits];
  net.wGrads       = new WGrads      [net.batchSize*net.numUnits];
  net.backGrads    = new float       [net.batchSize*net.numUnits];
  net.step         = new float       [net.batchSize*net.numUnits*net.unitSize];
}

void initUnit(Net& net, int l, int i) {
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  float *unit = getUnit(net, l, i);
  for (int x = 0; x < net.res; x++)
    for (int y = 0; y < net.res; y++)
      unit[I_unit(net, x, y)] = distribution(generator);
}

void initNet(Net& net) {
  for (int i = 0; i < net.depth; i++)
    for (int j = 0; j < lenLayer(net, i); j++)
      initUnit(net, i, j);
}

Net makeNet(
    int dim, int res, float reg,
    float **inputs, float *outputs, int numExamples,
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
  net.outputs     = outputs;
  net.numExamples = numExamples;
  net.rate        = rate;
  net.momentum    = momentum;
  net.batchSize   = batchSize;
  net.batchIndex  = numExamples/batchSize; // Past last batch. Triggers shuffle

  // Allocation
  allocNet(net);
  initNet (net);

  net.output = getActs(net, net.depth - 1);

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

void computeCPtLocs(Net& net, int l) {
  // Figure out where the previous layer of `acts` and `cPtLocs` is in memory
  CPtLocs *cPtLocs = getCPtLocs(net, l);
  float   *acts    = getActs   (net, l-1);

  // Compute act locs
  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    binPairs(
        net,
        acts[2*i+0], acts[2*i+1],
        cPtLocs[i].xbl, cPtLocs[i].xbu, cPtLocs[i].ybl, cPtLocs[i].ybu
    );
  }
}

void computeCPtDists(Net& net, int l) {
  CPtLocs  *cPtLocs  = getCPtLocs (net, l);
  CPtDists *cPtDists = getCPtDists(net, l);
  float    *acts     = getActs    (net, l-1);

  // Compute `cPtDists`
  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    distPairs(
        net,
        acts[2*i+0], acts[2*i+1],
        cPtLocs [i].xbl, cPtLocs [i].xbu, cPtLocs [i].ybl, cPtLocs [i].ybu,
        cPtDists[i].xdl, cPtDists[i].xdu, cPtDists[i].ydl, cPtDists[i].ydu
    );
  }
}

void computeCPtVals(Net& net, int l) {
  CPtLocs *cPtLocs  = getCPtLocs(net, l);
  CPtVals *cPtVals  = getCPtVals(net, l);
  float   *unit     = getUnit   (net, l);

  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    int xbl = cPtLocs[i].xbl;
    int xbu = cPtLocs[i].xbu;
    int ybl = cPtLocs[i].ybl;
    int ybu = cPtLocs[i].ybu;

    cPtVals[i].xlyl = unit[I_unit(net, xbl, ybl)];
    cPtVals[i].xuyl = unit[I_unit(net, xbu, ybl)];
    cPtVals[i].xlyu = unit[I_unit(net, xbl, ybu)];
    cPtVals[i].xuyu = unit[I_unit(net, xbu, ybu)];

    unit += net.unitSize;
  }
}

void computeActs(Net& net, int l) {
  CPtVals  *cPtVals  = getCPtVals (net, l);
  CPtDists *cPtDists = getCPtDists(net, l);
  float    *acts     = getActs    (net, l);
  float    norm      = (net.res - 1) * (net.res - 1);

  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    acts[i] = norm * (
        cPtVals[i].xlyl * cPtDists[i].xdu * cPtDists[i].ydu +
        cPtVals[i].xuyl * cPtDists[i].xdl * cPtDists[i].ydu +
        cPtVals[i].xlyu * cPtDists[i].xdu * cPtDists[i].ydl +
        cPtVals[i].xuyu * cPtDists[i].xdl * cPtDists[i].ydl
    );
  }
}

void forward(Net& net) {
  for (int i(0), I(net.depth); i < I; i++) {
    computeCPtLocs (net, i);
    computeCPtVals (net, i);
    computeCPtDists(net, i);
    computeActs    (net, i);
  }
}

void forward(Net& net, float* input) {
  net.input = input;
  forward(net);
}

/**************************** GRADIENT COMPUTATION ****************************/

void computeXGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  CPtVals  *cPtVals  = net.cPtVals;
  int       c        = (net.res - 1) * (net.res - 1);

  for (int i(0), I(net.numUnits); i < I; i++) {
    net.xGrads[i] = c * (
        cPtDists[i].ydu * (cPtVals[i].xuyl - cPtVals[i].xlyl) +
        cPtDists[i].ydl * (cPtVals[i].xuyu - cPtVals[i].xlyu)
    );
  }
}

void computeYGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  CPtVals  *cPtVals  = net.cPtVals;
  int       c        = (net.res - 1) * (net.res - 1);

  for (int i(0), I(net.numUnits); i < I; i++) {
    net.yGrads[i] = c * (
        cPtDists[i].xdu * (cPtVals[i].xlyu - cPtVals[i].xlyl) +
        cPtDists[i].xdl * (cPtVals[i].xuyu - cPtVals[i].xuyl)
    );
  }
}

void computeWGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  WGrads   *wGrads   = net.wGrads;
  float    norm      = (net.res - 1) * (net.res - 1);

  for (int i(0), I(net.numUnits); i < I; i++) {
    wGrads[i].xlyl = norm * cPtDists[i].xdu * cPtDists[i].ydu;
    wGrads[i].xlyu = norm * cPtDists[i].xdu * cPtDists[i].ydl;
    wGrads[i].xuyl = norm * cPtDists[i].xdl * cPtDists[i].ydu;
    wGrads[i].xuyu = norm * cPtDists[i].xdl * cPtDists[i].ydl;
  }
}

float dError(float output, float target) {
  return output - target;
}

void computeBackGrads(Net& net, float target) {
  net.backGrads[net.numUnits - 1] = 1.0;
  for (int i = net.depth - 1; i >= 1; i--) {
    int L = lenLayer(net, i);

    float *xGrads        = getXGrads   (net, i);
    float *yGrads        = getYGrads   (net, i);
    float *prevBackGrads = getBackGrads(net, i-1);
    float *nextBackGrads = getBackGrads(net, i);

    for (int j = L-1; j >= 0; j--) {
      prevBackGrads[2*j+0] = /*sgn*/(nextBackGrads[j]) * xGrads[j];
      prevBackGrads[2*j+1] = /*sgn*/(nextBackGrads[j]) * yGrads[j];
    }
  }

  float dE = dError(*net.output, target);
  for (int i(0), I(net.numUnits); i < I; i++)
      net.backGrads[i] *= dE;
}

void computeExampleGrads(Net& net) {
  ExampleGrads *exampleGrads = net.exampleGrads;
  float        *backGrads    = net.backGrads;
  WGrads       *wGrads       = net.wGrads;

  for (int i(0), I(net.numUnits); i < I; i++) {
    exampleGrads[i].xlyl = backGrads[i] * wGrads[i].xlyl;
    exampleGrads[i].xlyu = backGrads[i] * wGrads[i].xlyu;
    exampleGrads[i].xuyl = backGrads[i] * wGrads[i].xuyl;
    exampleGrads[i].xuyu = backGrads[i] * wGrads[i].xuyu;
  }
}

void backward(Net& net, float target) {
  computeXGrads      (net);
  computeYGrads      (net);
  computeWGrads      (net);
  computeBackGrads   (net, target);
  computeExampleGrads(net);
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
  {
    float reg = net.reg;
    for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
      net.regGrads[i] *= reg;
  }
}

void addExampleGrads(Net& net) {
  int           batchSize    = net.batchSize;
  float        *batchGrads   = net.batchGrads;
  ExampleGrads *exampleGrads = net.exampleGrads;
  CPtLocs      *cPtLocs      = net.cPtLocs;

  for (int i(0), I(net.numUnits); i < I; i++) {
    int xbl = cPtLocs[i].xbl;
    int xbu = cPtLocs[i].xbu;
    int ybl = cPtLocs[i].ybl;
    int ybu = cPtLocs[i].ybu;

    batchGrads[I_unit(net, xbl, ybl)] += exampleGrads[i].xlyl / batchSize;
    batchGrads[I_unit(net, xbu, ybl)] += exampleGrads[i].xuyl / batchSize;
    batchGrads[I_unit(net, xbl, ybu)] += exampleGrads[i].xlyu / batchSize;
    batchGrads[I_unit(net, xbu, ybu)] += exampleGrads[i].xuyu / batchSize;

    batchGrads += net.unitSize;
  }
}

void addExampleError(Net& net, float target) {
  float dErr = *net.output - target;
  net.batchMse += dErr * dErr / net.batchSize;
}

void normaliseContrast(Net& net) {
  for (int i(0), I(net.numUnits - 1); i < I; i++) {
    float *unit = &net.params[i*net.unitSize];

    float unitAvg;
    float unitVar;
    float eps = 1e-5;
    moments(unit, net.unitSize, unitAvg, unitVar);

    add(unit, net.unitSize, - unitAvg);
    if (unitVar > FLT_MIN + eps)
      mul(unit, net.unitSize, 0.25 / sqrt(unitVar + eps));
    add(unit, net.unitSize, + 0.5);
  }
}

void shuffleExamples(Net& net) {
  for (int i = 0, n(net.numExamples - 1); i < n; i++) {
    std::uniform_int_distribution<int> distribution(i, n - 1);
    int j = distribution(generator);

    std::swap(net.inputs [i], net.inputs [j]);
    std::swap(net.outputs[i], net.outputs[j]);
  }
}

void initBatchVars(Net& net) {
  // Shuffle examples
  bool epochDone = net.batchSize * (net.batchIndex + 1) > net.numExamples;
  if (epochDone) {
    shuffleExamples(net);
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
  int i = net.batchSize * (net.batchIndex + 0);
  int I = net.batchSize * (net.batchIndex + 1);
  net.batchIndex++;

  for ( ; i < I; i++) {
    forward (net, net.inputs[i]);
    backward(net, net.outputs[i]);
    addExampleGrads(net);
    addExampleError(net, net.outputs[i]);
  }
}

void sgd(Net& net) {
  computeRegGrads(net);
  computeBatchGrads(net);

  // Compute momentum
  for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    net.step[i] = net.momentum * net.step[i] -
                  net.rate * (net.batchGrads[i] + net.regGrads[i]);

  // Update params
  for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    net.params[i] += net.step[i];

  // TODO: Is this needed any more?
  //// Retard momentum
  //for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
    //if (net.params[i] < 0.0 || net.params[i] > 1.0)
      //net.step[i] = 0.0;

  //// Clamp params
  //for (int i(0), I(net.numUnits * net.unitSize); i < I; i++)
      //net.params[i] = clamp(net.params[i], 0.0, 1.0);
}

/******************************** BENCHMARKING ********************************/

float classificationError(Net& net, float* input, float target) {
  forward(net, input);

  if (roundf(*net.output) == target) return 0.0;
  return 1.0;
}

float classificationError(
    Net& net,
    float** inputs,
    float* outputs,
    int numExamples,
    int sampleSize=5000
) {
  sampleSize = min(sampleSize, numExamples);

  float sum = 0.0;
  for (int i = 0; i < sampleSize; i++) {
    std::uniform_int_distribution<int> distribution(0, sampleSize-1);
    int j = distribution(generator);

    sum += classificationError(net, inputs[j], outputs[j]) / sampleSize;
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
  //if (max(input, dim) == input[2]) return 1.0;

  if (numOnes(input, dim) % 2) return 1.0;

  return 0.0;
}

void makeData(float **&inputs, float *&outputs, int dim, int numExamples) {
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Allocate inputs
  inputs = new float*[numExamples];
  for (int i = 0; i < numExamples; i++)
    inputs[i] = new float[dim];

  // Allocate outputs
  outputs = new float[numExamples];

  // Init inputs
  for (int i = 0; i < numExamples; i++)
    for(int j = 0; j < dim; j++)
      inputs[i][j] = distribution(generator);

  // Init outputs
  for (int i = 0; i < numExamples; i++)
    outputs[i] = trueClassifier(inputs[i], dim);
}

/************************************ MAIN ************************************/

int main() {
  // MODEL VARS
  int   dim = 4;
  int   res = 10;
  float reg = 0.1;

  // OPTIMISER VARS
  float rate      = 0.1;
  float momentum  = 0.9;
  int   batchSize = 100;

  // MAKE INPUT RE-ORDERING MAP
  //float *map = makeQuasiConvMap(dim);

  // LOAD MNIST TRAINING SET
  //int     digit = 0;
  int     numExamplesTrn;
  float** inputsTrn;
  float*  outputsTrn;
  //loadMnist(inputsTrn, outputsTrn, numExamplesTrn, digit, "train", map);

  // LOAD MNIST TESTING SET
  //int     numExamplesTst;
  //float** inputsTst;
  //float*  outputsTst;
  //loadMnist(inputsTst, outputsTst, numExamplesTst, digit, "t10k", map);

  // LOAD SYNTHETIC DATA SET
  numExamplesTrn = 100000;
  makeData(inputsTrn, outputsTrn, dim, numExamplesTrn);

  // Make model
  Net net = makeNet(
      dim, res, reg,
      inputsTrn, outputsTrn, numExamplesTrn,
      rate, momentum, batchSize
  );

  // BENCHMARK
  //{
    //for (int i = 0; i < 5000; i++) {
      //forward(net, inputsTrn[rand()%numExamplesTrn]);
    //}
    //return 0;
  //}

  // OPTIMISE
  for (int i = 0; i < 10000; i++) {
    for (int j = 0; j < 10000; j++)
      sgd(net);
    net.reg *= 0.9;

    float e;
    e = classificationError(net, inputsTrn, outputsTrn, 10000, 10000);
    e *= 100;
    std::cout << "Train error (%): " << e << std::endl;

    //e = classificationError(net, inputsTst, outputsTst, numExamplesTst, numExamplesTst);
    //e *= 100;
    //std::cout << "Test error (%):  " << e << std::endl;

    std::cout << net.batchMse << std::endl;
    std::cout << std::endl;

    printParams(i, net);
  }
}
