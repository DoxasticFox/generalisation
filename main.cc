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

struct ExampleGrad { float xlyl; float xuyl; float xlyu; float xuyu; };
struct WGrads      { float xlyl; float xuyl; float xlyu; float xuyu; };
struct CPtLocs     { int   xbl;  int   xbu;  int   ybl;  int   ybu;  };
struct CPts        { float xlyl; float xuyl; float xlyu; float xuyu; };
struct CPtDists    { float xdl;  float xdu;  float ydl;  float ydu;  };

struct Net {
  int    dim; // Dimensionality of input
  int    res;
  int    reg;
  int    depth;
  int    numUnits;

  float       *batchGrads;
  ExampleGrad *exampleGrads;
  float       *xGrads;
  float       *yGrads;
  WGrads      *wGrads;
  float       *momentum;
  float       *params;

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

/*************************** NETWORK INITIALISATION ***************************/

void allocNet(Net& net) {
  net.batchGrads   = new float      [net.numUnits*net.res*net.res];
  net.exampleGrads = new ExampleGrad[net.numUnits];
  net.xGrads       = new float      [net.numUnits];
  net.yGrads       = new float      [net.numUnits];
  net.wGrads       = new wGrads     [net.numUnits];
  net.momentum     = new float      [net.numUnits*net.res*net.res];
  net.params       = new float      [net.numUnits*net.res*net.res];

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

  net.dim      = dim;
  net.res      = res;
  net.reg      = reg;
  net.depth    = ceil(log2(dim));
  net.numUnits = dim - 1;

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
  return;
}

void computeYGrads(Net& net) {
  return;
}

void computeWGrads(Net& net) {
  return;
}

void computeExampleGrads(Net& net) {
  return;
}

void backward(Net& net, float* target) {
  computeXGrads(net);
  computeYGrads(net);
  computeWGrads(net);
  computeExampleGrads(net);
}

//void computeGradient(Net& net, float* input, float* target) {
  //int m, n;

  //forward (net, input);
  //backward(net, target);


  //std::cout << "computeGradient [acts]" << std::endl;
  //for (int i = 0; i <= net.numLayers; i++)
  //printVector(net, net.acts, i);
  //std::cout << std::endl;

  //std::cout << "computeGradient [weights]" << std::endl;
  //for (int i = 0; i < net.numLayers; i++)
  //printMatrix(net, net.weights, i);

  //std::cout << "computeGradient [deltas]" << std::endl;
  //for (int i = 0; i <= net.numLayers; i++)
  //printVector(net, net.deltas, i);
  //std::cout << std::endl;

  //std::cout << "computeGradient [gradients]" << std::endl;
  //for (int i = 0; i < net.numLayers; i++)
  //printMatrix(net, net.gradients, i);
//}

/******************************** BENCHMARKING ********************************/

float classificationError(Net& net, float* input, float* target) {
  forward(net, input);

  //int   bestIndex = 0;
  //float bestDist  = FLT_MAX;
  //for (int i = 0; i < net.dimOutput; i++) {
    //float thisDist = fabs(net.output[i] - 1.0);
    //if (thisDist < bestDist) {
      //bestDist = thisDist;
      //bestIndex = i;
    //}
  //}

  //if (fabs(target[bestIndex] - 1.0) < FLT_EPSILON)
    //return 0.0;
  //else
    //return 100.0;
  return 1.0;
}

float classificationError(Net& net, float** inputs, float** targets, int numExamples) {
  float sum = 0.0;
  for (int i = 0; i < numExamples; i++)
    sum += classificationError(net, inputs[i], targets[i]) / numExamples;
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

/*********************** NUMERICAL GRADIENT COMPUTATION ***********************/

void computeGradientNumerically(Net& net, float* input, float* target) {
  //int m, n;
  //float h = 1e-3;

  //for (int i = 0; i < net.numLayers; i++){
    //for (int j = 0; j < m*n; j++) {
      //float addObj, subObj;

      //float prev = net.weights[i][j];
      //net.weights[i][j] = prev + h; addObj = obj(net, input, target);
      //net.weights[i][j] = prev - h; subObj = obj(net, input, target);
      //net.weights[i][j] = prev;

      //net.gradients[i][j] = (addObj - subObj) / (2.0 * h);
    //}
  //}

  //std::cout << "computeGradientNumerically [acts]" << std::endl;
  //for (int i = 0; i <= net.numLayers; i++)
  //printVector(net, net.acts, i);
  //std::cout << std::endl;

  //std::cout << "computeGradientNumerically [weights]" << std::endl;
  //for (int i = 0; i < net.numLayers; i++)
  //printMatrix(net, net.weights, i);

  //std::cout << "computeGradientNumerically [deltas]" << std::endl;
  //for (int i = 0; i <= net.numLayers; i++)
  //printVector(net, net.deltas, i);
  //std::cout << std::endl;

  //std::cout << "computeGradientNumerically [gradients]" << std::endl;
  //for (int i = 0; i < net.numLayers; i++)
  //printMatrix(net, net.gradients, i);
}

/********************************** LEARNING **********************************/

/* Performs a single weight update based on the example given by the
 * (`input`, `target`) pair.
 */
void sgd(
    Net& net,
    float* input, float* target,
    float rate,
    float momentum
) {
  //int m, n;
  //computeGradient(net, input, target);

  // 1. Compute weight update

  // 3. Update weights
}


void sgd(
    Net& net,
    float** inputs, float** targets,
    int numExamples,
    int batchSize,
    float rate=1.0,
    float momentum=0.9
) {
  std::uniform_int_distribution<int> distribution(0, numExamples-1);

  for (int i = 0; i < batchSize; i++) {
    int j = distribution(generator);
    sgd(net, inputs[j], targets[j], rate, momentum);
  }
}

/************************************ MAIN ************************************/

int main() {
  // OPTIMISER VARS
  float rate      = 1.0;
  float momentum  = 0.9;
  int   batchSize = 100;

  // MODEL VARS
  int   dim = 128*128;
  int   res = 10;
  float reg = 0.001;
  Net net = makeNet(dim, res, reg);

  // FEED-FORWARD TEST
  float* input = new float[dim];
  input[0] = 0.0;
  input[1] = 0.0;
  input[2] = 0.1;
  input[3] = 0.2;

  forward(net, input);
  std::cout << *net.output << std::endl;
  for (int i = 0; i < 5000; i++) {
    forward(net, input);
  }
  std::cout << *net.output << std::endl;

  //std::cout << "CPtLocs:" << std::endl;
  //for (int i = 0; i < net.numUnits*4; i++) {
    //std::cout << net.cPtLocs[i] << " ";
    //if ((i + 1) % 4 == 0) std:: cout << "  ";
  //}
  //std::cout << std::endl;

  //std::cout << "CPtDists:" << std::endl;
  //for (int i = 0; i < net.numUnits*4; i++) {
    //std::cout << net.cPtDists[i] << " ";
    //if ((i + 1) % 4 == 0) std:: cout << "  ";
  //}
  //std::cout << std::endl;

  //std::cout << "Acts:" << std::endl;
  //for (int i = 0; i < net.numUnits; i++)
    //std::cout << net.acts[i] << " ";
  //std::cout << std::endl;

  return 0;

  // LOAD MNIST
  int numExamples;
  float** inputs;
  float** targets;

  loadMnist(inputs, targets, numExamples);

  sgd(net, inputs, targets, numExamples, batchSize, rate, momentum);
}
