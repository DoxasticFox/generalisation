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

struct Net {
  int    dim; // Dimensionality of input
  int    res;
  int    reg;
  int    depth;
  int    numUnits;

  float *batchGrads;
  float *exampleGrads;
  float *momentum;
  float *params;

  float *input;
  float *output;

  int   *actLocs;
  float *actDists;
  float *acts;
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

// Indexes the elements (units) in a layer
int I_layer(Net& net, int i) {
  return (net.res * net.res) * i;
}

// Indexes the elements (pixels) in a unit
int I_unit(Net& net, int x, int y) {
  return net.res * x + y;
}

int I_acts(Net& net, int l, int i) {
  return numUnitsAbove(net, l, i);
}

int I_acts(Net& net, int l) {
  return I_acts(net, l, 0);
}

int I_actLocs(Net& net, int l, int i) {
  return numUnitsAbove(net, l, i) * 4;
}

int I_actLocs(Net& net, int l) {
  return I_actLocs(net, l, 0);
}

int I_actDists(Net& net, int l, int i) {
  return numUnitsAbove(net, l, i) * 4;
}

int I_actDists(Net& net, int l) {
  return numUnitsAbove(net, l, 0) * 4;
}

int I_params(Net& net, int l, int i, int x, int y) {
  return (net.res * net.res) * numUnitsAbove(net, l, i) + I_unit(net, x, y);
}

int I_params(Net& net, int l, int i) {
  return I_params(net, l, i, 0, 0);
}

int I_params(Net& net, int l) {
  return I_params(net, l, 0);
}

/********************************** GETTERS ***********************************/

float* getLayer(Net& net, int l) {
  return &net.params[I_params(net, l)];
}

float* getUnit(Net& net, int l, int i) {
  return &net.params[I_params(net, l, i)];
}

float getParam(Net& net, int l, int i, int x, int y) {
  return net.params[I_params(net, l, i, x, y)];
}

float* getActs(Net& net, int l, int i) {
  if (l < 0) return &net.input[I_acts(net, 0, i)];
  else       return &net.acts [I_acts(net, l, i)];
}

float* getActs(Net& net, int l) {
  return getActs(net, l, 0);
}

int* getActLocs(Net& net, int l, int i) {
  return &net.actLocs[I_actLocs(net, l, i)];
}

int* getActLocs(Net& net, int l) {
  return getActLocs(net, l, 0);
}

float* getActDists(Net& net, int l, int i) {
  return &net.actDists[I_actDists(net, l, i)];
}

float* getActDists(Net& net, int l) {
  return getActDists(net, l, 0);
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
  net.batchGrads   = new float[net.numUnits*net.res*net.res];
  net.exampleGrads = new float[net.numUnits*4];
  net.momentum     = new float[net.numUnits*net.res*net.res];
  net.params       = new float[net.numUnits*net.res*net.res];

  net.acts         = new float[net.numUnits];
  net.actLocs      = new   int[net.numUnits*4];
  net.actDists     = new float[net.numUnits*4];
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

void computeActLocs(Net& net, int l) {
  // Figure out where the previous layer of `acts` and `actLocs` is in memory
  int   *actLocs = getActLocs(net, l);
  float *acts    = getActs   (net, l-1);

  // Compute act locs
  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
    float x = acts[0];
    float y = acts[1];

    int xbl, xbu, ybl, ybu;
    binPairs(net, x, y, xbl, xbu, ybl, ybu);

    actLocs[0] = xbl;
    actLocs[1] = xbu;
    actLocs[2] = ybl;
    actLocs[3] = ybu;

    acts    += 2;
    actLocs += 4;
  }
}

void computeActDists(Net& net, int l) {
  int   *actLocs  = getActLocs (net, l);
  float *actDists = getActDists(net, l);
  float *acts     = getActs    (net, l-1);

  // Compute `actDists`
  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
    float x = acts[0];
    float y = acts[1];

    int xbl = actLocs[0];
    int xbu = actLocs[1];
    int ybl = actLocs[2];
    int ybu = actLocs[3];

    float xdl, xdu, ydl, ydu;
    distPairs(
        net, x, y,
        xbl, xbu, ybl, ybu,
        xdl, xdu, ydl, ydu
    );

    actDists[0] = xdl;
    actDists[1] = xdu;
    actDists[2] = ydl;
    actDists[3] = ydu;

    acts     += 2;
    actLocs  += 4;
    actDists += 4;
  }
}

void computeActs(Net& net, int l) {
  int   *actLocs  = getActLocs (net, l, 0);
  float *actDists = getActDists(net, l, 0);
  float *acts     = getActs    (net, l, 0);
  float *unit     = getUnit    (net, l, 0);

  int L = lenLayer(net, l);
  for (int i = 0; i < L; i++) {
    int xbl = actLocs[0];
    int xbu = actLocs[1];
    int ybl = actLocs[2];
    int ybu = actLocs[3];

    float xdl = actDists[0];
    float xdu = actDists[1];
    float ydl = actDists[2];
    float ydu = actDists[3];

    *acts = (net.res - 1) * (net.res - 1) * (
        unit[I_unit(net, xbl, ybl)] * xdu * ydu +
        unit[I_unit(net, xbu, ybl)] * xdl * ydu +
        unit[I_unit(net, xbl, ybu)] * xdu * ydl +
        unit[I_unit(net, xbu, ybu)] * xdl * ydl
    );

    actLocs  += 4;
    actDists += 4;
    acts     += 1;
    unit     += net.res * net.res;
  }
}

void forward(Net& net) {
  for (int i = 0; i < net.depth; i++) {
    computeActLocs (net, i);
    computeActDists(net, i);
    computeActs    (net, i);
  }
}

void forward(Net& net, float* input) {
  net.input = input;
  forward(net);
}

/**************************** GRADIENT COMPUTATION ****************************/

void backward(Net& net, float* target) {
  return;
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
  int   dim = 4;
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
  //for (int i = 0; i < 5000; i++) {
    //forward(net, input);
    //std::cout << *net.output << std::endl;
  //}

  //std::cout << "ActLocs:" << std::endl;
  //for (int i = 0; i < net.numUnits*4; i++) {
    //std::cout << net.actLocs[i] << " ";
    //if ((i + 1) % 4 == 0) std:: cout << "  ";
  //}
  //std::cout << std::endl;

  //std::cout << "ActDists:" << std::endl;
  //for (int i = 0; i < net.numUnits*4; i++) {
    //std::cout << net.actDists[i] << " ";
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
