#include <iostream>
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

  float *actLocs;
  float *actDists;
  float *acts;
};

std::default_random_engine generator;

/***************************** COUNTING FUNCTIONS *****************************/

int lenLayer(Net& net, int l) {
  return pow2(net.depth - l - 1);
}

// Number of units above the layer indexed by `l`
int numUnitsAbove(Net& net, int l) {
  return net.dim - pow2(net.depth - l);
}

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

float getPixel(Net& net, int l, int i, int x, int y) {
  return net.params[I_params(net, l, i, x, y)];
}

float* getActs(Net& net, int l, int i) {
  return &net.acts[I_acts(net, l, i)];
}

float* getActs(Net& net, int l) {
  return getActs(net, l, 0);
}

float* getActLocs(Net& net, int l, int i) {
  return &net.actLocs[I_actLocs(net, l, i)];
}

float* getActLocs(Net& net, int l) {
  return getActLocs(net, l, 0);
}

float* getActDists(Net& net, int l, int i) {
  return &net.actDists[I_actDists(net, l, i)];
}

float* getActDists(Net& net, int l) {
  return getActDists(net, l, 0);
}

/*************************** NETWORK INITIALISATION ***************************/

void allocNet(Net& net) {
  net.batchGrads   = new float[net.numUnits*net.res*net.res];
  net.exampleGrads = new float[net.numUnits*4];
  net.momentum     = new float[net.numUnits*net.res*net.res];
  net.params       = new float[net.numUnits*net.res*net.res];

  net.acts         = new float[net.numUnits];
  net.actLocs      = new float[net.numUnits*4];
  net.actDists     = new float[net.numUnits*4];
}

void initUnit(Net& net, int l, int i) {
  // Flip a coin to pick between abs1 and abs2
  bool abs1 = rand() % 2 == 0;
  bool abs2 = !abs1;

  float *unit = getUnit(net, l, i);
  for (int x = 0; x < net.res; x++) {
    for (int y = 0; y < net.res; y++) {
      int I = I_unit(net, x, y);
      if (abs1) unit[I] = fabs((x + y)/(net.res - 1.0) - 1.0);
      if (abs2) unit[I] = fabs((x - y)/(net.res - 1.0)      );
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
    float& xbl, float& xbu
) {
  int resSub    = net.res - 1;
  int resSubSub = net.res - 2;

  xbl = (int) (x * resSub);
  if (xbl >= resSub) xbl = resSubSub;

  xbu = xbl + 1;
}

void binPairs(
    Net& net, float x, float y,
    float& xbl, float& xbu, float& ybl, float& ybu
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
  float *actLocs  = getActLocs(net, l);
  float *acts;
  if (l == 0) acts = net.input;
  else        acts = getActs(net, l-1);

  // Compute act locs
  for (int i = 0; i < lenLayer(net, l); i++) {
    float x = acts[i*2+0];
    float y = acts[i*2+1];

    float xbl, xbu, ybl, ybu;
    binPairs(net, x, y, xbl, xbu, ybl, ybu);

    actLocs[i*4+0] = xbl;
    actLocs[i*4+1] = xbu;
    actLocs[i*4+2] = ybl;
    actLocs[i*4+3] = ybu;
  }
}

void computeActDists(Net& net, int l) {
  float *actLocs  = getActLocs (net, l);
  float *actDists = getActDists(net, l);
  float *acts;
  if (l == 0) acts = net.input;
  else        acts = getActs(net, l-1);

  // Compute `actDists`
  for (int i = 0; i < lenLayer(net, l); i++) {
    float x = acts[i*2+0];
    float y = acts[i*2+1];

    float xbl = actLocs[i*4+0];
    float xbu = actLocs[i*4+1];
    float ybl = actLocs[i*4+2];
    float ybu = actLocs[i*4+3];

    float xdl, xdu, ydl, ydu;
    distPairs(
        net, x, y,
        xbl, xbu, ybl, ybu,
        xdl, xdu, ydl, ydu
    );

    actDists[i*4+0] = xdl;
    actDists[i*4+1] = xdu;
    actDists[i*4+2] = ydl;
    actDists[i*4+3] = ydu;
  }
}

void computeActs(Net& net, int l) {
  float *actLocs  = getActLocs (net, l, 0);
  float *actDists = getActDists(net, l, 0);
  float *acts     = getActs    (net, l, 0);
  float *unit     = getUnit    (net, l, 0);

  for (int i = 0; i < lenLayer(net, l); i++) {
    float xbl = actLocs[i*4+0];
    float xbu = actLocs[i*4+1];
    float ybl = actLocs[i*4+2];
    float ybu = actLocs[i*4+3];

    float ydl = actDists[i*4+0];
    float ydu = actDists[i*4+1];
    float xdl = actDists[i*4+2];
    float xdu = actDists[i*4+3];

    float resSub = net.res - 1;

    acts[i] = resSub * resSub * (
        unit[I_unit(net, xbl, ybl)] * xdu * ydu +
        unit[I_unit(net, xbu, ybl)] * xdl * ydu +
        unit[I_unit(net, xbl, ybu)] * xdu * ydl +
        unit[I_unit(net, xbu, ybu)] * xdl * ydl
    );
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

float act(float x) {
  return max(0, x);
  return (x > 0.0) ? 1.0 : 0.0; // a = 1.0;
  return clamp(x, 0.0, 1.0);
  return fabs(x);
  return x;
}

float gradAct(float x) {
  // relu(x)
  if (x >= 0.0) return 1.0;
  else          return 0.0;

  // hard-tanh(x)
  if (x >= -1.0 && x <= +1.0) return 1.0;
  else                        return 0.0;

  // bernoulli(x), hard-sig(x)
  if (x >= 0.0 && x <= 1.0) return 1.0;
  else                      return 0.0;

  // abs(x)
  if (x > 0.0) return +1.0;
  if (x < 0.0) return -1.0;
               return  0.0;

  // iden(x)
  return 1.0;
}

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
  // TRAINER VARS
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
  //for (int i = 0; i < dim; i++)
    //input[i] = 1.0;
  //input[0] = 0.3;
  //input[1] = 0.2;
  //input[2] = 0.9;
  //input[3] = 0.1;

  for (int i = 0; i < 10000; i++) {
    forward(net, input);
  }
  std::cout << *net.output << std::endl;

  return 0;

  // LOAD MNIST
  int numExamples;
  float** inputs;
  float** targets;

  loadMnist(inputs, targets, numExamples);

  sgd(net, inputs, targets, numExamples, batchSize, rate, momentum);

  //// TRAIN
  //for (int i = 0; i < 100000; i++) {
    //sgd(net, inputs, targets, numExamples, batchSize, rate, momentum);
    //std::cout << "Error (train, MSE): " << obj(net, inputs, targets, 400) << "\t";
    //std::cout << "Error (train,   %): " << classificationError(net, inputs, targets, 400);
    //std::cout << std::endl;
    //std::cout << "Error (test,  MSE): " << obj(net, &inputs[49999], &targets[49999], 400) << "\t";
    //std::cout << "Error (test,    %): " << classificationError(net, &inputs[49999], &targets[49999], 400);
    //std::cout << std::endl;

    //// PRINT OUTPUTS
    //std::cout << std::endl << "outputs" << std::endl;
    //for (int i = 0; i < 10; i++) {
      //forward(net, inputs[i]);
      //printVector(net.output, net.dimOutput);
    //}
  //}


  //// PRINT INPUTS
  //std::cout << std::endl << "inputs" << std::endl;
  //for (int i = 0; i < 10; i++)
    //printMatrix(inputs[i], 28, 28);

  //// PRINT TARGETS
  //std::cout << std::endl << "targets" << std::endl;
  //for (int i = 0; i < 10; i++)
    //printVector(targets[i], net.dimOutput);
}
