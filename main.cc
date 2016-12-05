#include <iostream>
#include <fstream>
#include <iomanip>
#include <climits>
#include <random>
#include <vector>
#include "math.h"
#include "float.h"
#include "mnist.h"
#include "more-math.h"

struct ExampleGrads { float xlyl; float xuyl; float xlyu; float xuyu; };
struct WGrads       { float xlyl; float xuyl; float xlyu; float xuyu; };
struct CPtLocs      { int   xbl;  int   xbu;  int   ybl;  int   ybu;  };
struct CPts         { float xlyl; float xuyl; float xlyu; float xuyu; };
struct CPtDists     { float xdl;  float xdu;  float ydl;  float ydu;  };

struct Net {
  int    dim; // Dimensionality of input
  int    res;
  float  reg;
  int    depth;
  int    numUnits;
  int    unitSize;

  float        *params;
  float        *momentum;
  float        *batchGrads;
  float        *xRegGrads;
  ExampleGrads *exampleGrads;
  float        *xGrads;
  float        *yGrads;
  WGrads       *wGrads;
  float        *backGrads;

  float *input;
  float *output;

  CPtLocs  *cPtLocs;
  CPts     *cPts;
  CPtDists *cPtDists;
  float    *acts;
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
  return (net.res * net.res) * I(net, l, i) + I_unit(net, x, y);
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

CPts* getCPts(Net& net, int l, int i) {
  return &net.cPts[I(net, l, i)];
}

CPts* getCPts(Net& net, int l) {
  return getCPts(net, l, 0);
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
  net.batchGrads   = new float       [net.numUnits*net.res*net.res];
  net.exampleGrads = new ExampleGrads[net.numUnits];
  net.xRegGrads    = new float       [net.numUnits*net.res*net.res];
  net.xGrads       = new float       [net.numUnits];
  net.yGrads       = new float       [net.numUnits];
  net.wGrads       = new WGrads      [net.numUnits];
  net.backGrads    = new float       [net.numUnits];
  net.momentum     = new float       [net.numUnits*net.res*net.res];
  net.params       = new float       [net.numUnits*net.res*net.res];

  net.cPtLocs      = new CPtLocs [net.numUnits];
  net.cPts         = new CPts    [net.numUnits];
  net.cPtDists     = new CPtDists[net.numUnits];
  net.acts         = new float   [net.numUnits];
}

float abs1(float x, float y) { return fabs(x + y - 1.0); }
float abs2(float x, float y) { return fabs(x - y      ); }

void initUnit(Net& net, int l, int i) {
  // Flip a coin to pick between abs1 and abs2
  bool a1 = rand() % 2 == 0;
  bool a2 = !a1;

  float *unit = getUnit(net, l, i);
  for (int x = 0; x < net.res; x++) {
    for (int y = 0; y < net.res; y++) {
      float fx = x/(net.res - 1.0);
      float fy = y/(net.res - 1.0);

      if (a1) unit[I_unit(net, x, y)] = abs1(fx, fy);
      if (a2) unit[I_unit(net, x, y)] = abs2(fx, fy);
    }
  }
}

void initNet(Net& net) {
  for (int i = 0; i < net.depth; i++)
    for (int j = 0; j < lenLayer(net, i); j++)
      initUnit(net, i, j);
}

Net makeNet(int dim, int res, float reg) {
  Net net;

  net.depth    = ceil(log2(dim));
  net.dim      = pow(2, net.depth);
  net.res      = res;
  net.reg      = reg;
  net.numUnits = net.dim - 1;
  net.unitSize = res * res;

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
  int resSub    = net.res - 1;
  int resSubSub = net.res - 2;

  xbl = (int) (x * resSub);
  if (xbl >= resSub) xbl = resSubSub;

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
  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
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
  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
    distPairs(
        net,
        acts[2*i+0], acts[2*i+1],
        cPtLocs [i].xbl, cPtLocs [i].xbu, cPtLocs [i].ybl, cPtLocs [i].ybu,
        cPtDists[i].xdl, cPtDists[i].xdu, cPtDists[i].ydl, cPtDists[i].ydu
    );
  }
}

void computeCPts(Net& net, int l) {
  CPtLocs *cPtLocs  = getCPtLocs(net, l);
  CPts    *cPts     = getCPts   (net, l);
  float   *unit     = getUnit   (net, l);

  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
    int xbl = cPtLocs[i].xbl;
    int xbu = cPtLocs[i].xbu;
    int ybl = cPtLocs[i].ybl;
    int ybu = cPtLocs[i].ybu;

    cPts[i].xlyl = unit[I_unit(net, xbl, ybl)];
    cPts[i].xuyl = unit[I_unit(net, xbu, ybl)];
    cPts[i].xlyu = unit[I_unit(net, xbl, ybu)];
    cPts[i].xuyu = unit[I_unit(net, xbu, ybu)];

    unit += net.res * net.res;
  }
}

void computeActs(Net& net, int l) {
  CPts     *cPts     = getCPts    (net, l);
  CPtDists *cPtDists = getCPtDists(net, l);
  float    *acts     = getActs    (net, l);

  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
    acts[i] = (net.res - 1) * (net.res - 1) * (
        cPts[i].xlyl * cPtDists[i].xdu * cPtDists[i].ydu +
        cPts[i].xuyl * cPtDists[i].xdl * cPtDists[i].ydu +
        cPts[i].xlyu * cPtDists[i].xdu * cPtDists[i].ydl +
        cPts[i].xuyu * cPtDists[i].xdl * cPtDists[i].ydl
    );
  }
}

void forward(Net& net) {
  for (int i = 0; i < net.depth; i++) {
    computeCPtLocs (net, i);
    computeCPts    (net, i);
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
  CPts     *cPts     = net.cPts;
  for (int i = 0; i < net.numUnits; i++) {
    net.xGrads[i] = (net.res - 1) * (net.res - 1) * (
        cPtDists[i].ydu * (cPts[i].xuyl - cPts[i].xlyl) +
        cPtDists[i].ydl * (cPts[i].xuyu - cPts[i].xlyu)
    );
  }
}

void computeYGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  CPts     *cPts     = net.cPts;
  for (int i = 0; i < net.numUnits; i++) {
    net.yGrads[i] = (net.res - 1) * (net.res - 1) * (
        cPtDists[i].xdu * (cPts[i].xlyu - cPts[i].xlyl) +
        cPtDists[i].xdl * (cPts[i].xuyu - cPts[i].xuyl)
    );
  }
}

void computeWGrads(Net& net) {
  CPtDists *cPtDists = net.cPtDists;
  WGrads   *wGrads   = net.wGrads;
  float    norm      = (net.res - 1) * (net.res - 1);
  for (int i = 0; i < net.numUnits; i++) {
    wGrads[i].xlyl = norm * cPtDists[i].xdu * cPtDists[i].ydu;
    wGrads[i].xlyu = norm * cPtDists[i].xdu * cPtDists[i].ydl;
    wGrads[i].xuyl = norm * cPtDists[i].xdl * cPtDists[i].ydu;
    wGrads[i].xuyu = norm * cPtDists[i].xdl * cPtDists[i].ydl;
  }
}

void computeBackGrads(Net& net, float target) {
  net.backGrads[net.numUnits - 1] = *net.output - target;

  for (int i = net.depth - 1; i >= 1; i--) {
    int L = lenLayer(net, i);

    float *xGrads        = getXGrads(net, i);
    float *yGrads        = getYGrads(net, i);
    float *prevBackGrads = getBackGrads(net, i-1);
    float *nextBackGrads = getBackGrads(net, i);

    for (int j = L-1; j >= 0; j--) {
      prevBackGrads[2*j+0] = nextBackGrads[j] * xGrads[j];
      prevBackGrads[2*j+1] = nextBackGrads[j] * yGrads[j];
    }
  }
}

void computeExampleGrads(Net& net) {
  ExampleGrads *exampleGrads = net.exampleGrads;
  float        *backGrads    = net.backGrads;
  WGrads       *wGrads       = net.wGrads;

  for (int i = 0; i < net.numUnits; i++) {
    exampleGrads[i].xlyl = backGrads[i] * wGrads[i].xlyl;
    exampleGrads[i].xlyu = backGrads[i] * wGrads[i].xlyu;
    exampleGrads[i].xuyl = backGrads[i] * wGrads[i].xuyl;
    exampleGrads[i].xuyu = backGrads[i] * wGrads[i].xuyu;
  }
}

void backward(Net& net, float target) {
  computeXGrads(net);
  computeYGrads(net);
  computeWGrads(net);
  computeBackGrads(net, target);
  computeExampleGrads(net);
}

// TODO: Profile function
void computeRegularisationGrads(Net& net) {
  float *unit;
  float *xRegGrads;
  float *batchGrads;

  batchGrads = net.batchGrads;
  for (int i = 0; i < net.numUnits * net.res * net.res; i++)
    batchGrads[i] = 0.0;

  // Compute regularisation along x direction
  unit      = net.params;
  xRegGrads = net.xRegGrads;
  for (int i = 0; i < net.numUnits; i++) {
    for (int x = 0; x < net.res; x++) {
      for (int y = 0; y < net.res; y++) {
        if (x == 0) {
          xRegGrads[I_unit(net, x, y)] = unit[I_unit(net, x+0, y)] +
                                         unit[I_unit(net, x+1, y)];
        } else if (x == net.res - 1) {
          xRegGrads[I_unit(net, x, y)] = unit[I_unit(net, x-1, y)] +
                                         unit[I_unit(net, x+0, y)];
        } else {
          xRegGrads[I_unit(net, x, y)] = unit[I_unit(net, x-1, y)] +
                                         unit[I_unit(net, x+0, y)] +
                                         unit[I_unit(net, x+1, y)];
        }
      }
    }
    unit      += net.res * net.res;
    xRegGrads += net.res * net.res;
  }

  // Compute regularisation along y direction
  xRegGrads  = net.xRegGrads;
  batchGrads = net.batchGrads;
  for (int i = 0; i < net.numUnits; i++) {
    for (int x = 0; x < net.res; x++) {
      for (int y = 0; y < net.res; y++) {
        if (y == 0) {
          batchGrads[I_unit(net, x, y)] = xRegGrads[I_unit(net, x, y+0)] +
                                          xRegGrads[I_unit(net, x, y+1)];
        } else if (y == net.res - 1) {
          batchGrads[I_unit(net, x, y)] = xRegGrads[I_unit(net, x, y-1)] +
                                          xRegGrads[I_unit(net, x, y+0)];
        } else {
          batchGrads[I_unit(net, x, y)] = xRegGrads[I_unit(net, x, y-1)] +
                                          xRegGrads[I_unit(net, x, y+0)] +
                                          xRegGrads[I_unit(net, x, y+1)];
        }
      }
    }
    xRegGrads  += net.res * net.res;
    batchGrads += net.res * net.res;
  }

  // Compute complete regularisation
  batchGrads = net.batchGrads;
  unit       = net.params;
  for (int i = 0; i < net.numUnits; i++) {
    for (int x = 0; x < net.res; x++) {
      for (int y = 0; y < net.res; y++) {
        bool isCorner = (x == 0 || x == net.res - 1) &&
                        (y == 0 || y == net.res - 1);
        bool isEdge   = (x == 0 || x == net.res - 1  ||
                         y == 0 || y == net.res - 1) &&
                        !isCorner;

        float c;
        if      (isCorner) c = 4.0;
        else if (isEdge)   c = 6.0;
        else               c = 9.0;

        batchGrads[I_unit(net, x, y)] = c * unit  [I_unit(net, x, y)] -
                                        batchGrads[I_unit(net, x, y)];
      }
    }
    unit       += net.res * net.res;
    batchGrads += net.res * net.res;
  }

  // Scale gradients by regularisation hyper-parameter
  batchGrads = net.batchGrads;
  for (int i = 0; i < net.numUnits * net.res * net.res; i++)
    batchGrads[i] *= net.reg;
}

void addExampleGrads(Net& net, float rate) {
  ExampleGrads *exampleGrads = net.exampleGrads;
  CPtLocs      *cPtLocs      = net.cPtLocs;
  float        *batchGrads   = net.batchGrads;

  for (int i = 0; i < net.numUnits; i++) {
    int xbl = cPtLocs[i].xbl;
    int xbu = cPtLocs[i].xbu;
    int ybl = cPtLocs[i].ybl;
    int ybu = cPtLocs[i].ybu;

    batchGrads[I_unit(net, xbl, ybl)] += rate * exampleGrads[i].xlyl;
    batchGrads[I_unit(net, xbu, ybl)] += rate * exampleGrads[i].xuyl;
    batchGrads[I_unit(net, xbl, ybu)] += rate * exampleGrads[i].xlyu;
    batchGrads[I_unit(net, xbu, ybu)] += rate * exampleGrads[i].xuyu;

    batchGrads += net.res * net.res;
  }
}

void normaliseContrast(Net& net) {
  for (int i = 0; i < net.numUnits - 1; i++) {
    float *unit = &net.params[i*net.unitSize];
    float unitMin = min(unit, net.unitSize);
    float unitMax = max(unit, net.unitSize);

    sub(unit, net.unitSize, unitMin);
    div(unit, net.unitSize, unitMax - unitMin);
  }
}

/* Performs a single weight update based on the example given by the
 * (`input`, `target`) pair.
 */
void sgd(
    Net& net,
    float* input, float target,
    float rate
) {
  forward (net, input);
  backward(net, target);
  addExampleGrads(net, rate);
}


void sgd(
    Net& net,
    float** inputs, float* targets,
    int numExamples,
    int batchSize,
    float rate=1.0,
    float momentum=0.9
) {
  // Compute gradient's regularisation component
  computeRegularisationGrads(net);

  // Compute gradient's batch component
  std::uniform_int_distribution<int> distribution(0, numExamples-1);
  for (int i = 0; i < batchSize; i++) {
    int j = distribution(generator);
    sgd(net, inputs[j], targets[j], 1.0/batchSize);
  }

  // Compute momentum
  for (int i = 0; i < net.numUnits * net.res * net.res; i++)
    net.momentum[i] = rate * net.batchGrads[i] + momentum * net.momentum[i];

  // Update params
  for (int i = 0; i < net.numUnits * net.res * net.res; i++)
    net.params[i] -= net.momentum[i];

  //// Retard momentum
  for (int i = 0; i < net.numUnits * net.res * net.res; i++)
    if (net.params[i] < 0.0 || net.params[i] > 1.0)
      net.momentum[i] = 0.0;

  // Clamp params
  for (int i = 0; i < net.numUnits * net.res * net.res; i++)
      net.params[i] = clamp(net.params[i], 0.0, 1.0);

  normaliseContrast(net);
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

float obj(Net& net, float* input, float* target) {
  forward(net, input);

  float sum = 0.0;
  //for (int i = 0; i < net.dimOutput; i++) {
    //float d = net.output[i] - target[i];
    //sum += d * d / 2.0;
  //}

  return sum;
}

float obj(Net& net, float** inputs, float** targets, int numExamples) {
  float sum = 0.0;
  for (int i = 0; i < numExamples; i++)
    sum += obj(net, inputs[i], targets[i]) / numExamples;
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
  if (max(input, dim) == input[2]) return 1.0;

  //if (numOnes(input, dim) % 2) return 1.0;

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
  int   dim = 1024;
  int   res = 5;
  float reg = 0.001;
  Net net = makeNet(dim, res, reg);

  // LOAD MNIST
  int     digit = 0;
  int     numExamples;
  float** inputs;
  float*  outputs;
  loadMnist(inputs, outputs, numExamples, digit);

  // OPTIMISER VARS
  float rate      = 1.0;
  float momentum  = 0.0;
  int   batchSize = 1000;

  // OPTIMISE
  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < 1760; i++) {
      sgd(net, inputs, outputs, numExamples, batchSize, rate, momentum);
    }

    forward(net, inputs[0]);
    std::cout << *net.output << std::endl;
    std::cout << "%" << classificationError(net, inputs, outputs, numExamples) * 100 << std::endl;
    printParams(j, net);
  }
}
