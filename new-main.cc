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

  float **batchInputs;
  float  *batchOutputs;

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
    for (int j(0), J(net.batchSize); j < J; j++) {
      binPairs(
          net,
          acts[(2*i+0)*J + j], acts[(2*i+1)*J + j],
          cPtLocs[i*J+j].xbl, cPtLocs[i*J+j].xbu, cPtLocs[i*J+j].ybl, cPtLocs[i*J+j].ybu
      );
    }
  }
}

void computeCPtDists(Net& net, int l) {
  CPtLocs  *cPtLocs  = getCPtLocs (net, l);
  CPtDists *cPtDists = getCPtDists(net, l);
  float    *acts     = getActs    (net, l-1);

  // Compute `cPtDists`
  for (int i(0), I(lenLayer(net, l)); i < I; i++) {
    for (int j(0), J(net.batchSize); j < J; j++) {
      distPairs(
          net,
          acts[(2*i+0)*J + j], acts[(2*i+1)*J + j],
          cPtLocs [i].xbl, cPtLocs [i].xbu, cPtLocs [i].ybl, cPtLocs [i].ybu,
          cPtDists[i].xdl, cPtDists[i].xdu, cPtDists[i].ydl, cPtDists[i].ydu
      );
    }
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

void batchNorm(net, i) {
  return;
}

void forward(Net& net) {
  for (int i(0), I(net.depth); i < I; i++) {
    computeCPtLocs (net, i);
    computeCPtVals (net, i);
    computeCPtDists(net, i);
    computeActs    (net, i);
    if (i < I - 1)
    batchNorm      (net, i);
  }
}

void forward(Net& net, float** batchInputs) {
  net.batchInputs = batchInputs; // TODO: You need to swap the indices on these
  forward(net);
}

/**************************** GRADIENT COMPUTATION ****************************/

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
  int i = net.batchSize * net.batchIndex;
  net.batchIndex++;

  forward (net, &net.inputs[i]);
  backward(net, &net.outputs[i]);

  //return;

  //// Get loop bounds and increment batchIndex for the next call
  //int i = net.batchSize * (net.batchIndex + 0);
  //int I = net.batchSize * (net.batchIndex + 1);
  //net.batchIndex++;

  //for ( ; i < I; i++) {
    //forward (net, net.inputs[i]);
    //backward(net, net.outputs[i]);
    //addExampleGrads(net);
    //addExampleError(net, net.outputs[i]);
  //}
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

  // LOAD SYNTHETIC DATA SET
  int     numExamplesTrn = 100000;
  float** inputsTrn;
  float*  outputsTrn;

  makeData(inputsTrn, outputsTrn, dim, numExamplesTrn);

  // Make model
  Net net = makeNet(
      dim, res, reg,
      inputsTrn, outputsTrn, numExamplesTrn,
      rate, momentum, batchSize
  );

  // OPTIMISE
  for (int i = 0; i < 10000; i++) {
    for (int j = 0; j < 10000; j++)
      sgd(net);
    net.reg *= 0.9;

    float e;
    e = classificationError(net, inputsTrn, outputsTrn, 10000, 10000);
    e *= 100;
    std::cout << "Train error (%): " << e << std::endl;

    std::cout << net.batchMse << std::endl;
    std::cout << std::endl;

    printParams(i, net);
  }
}
