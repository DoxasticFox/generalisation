#include <iostream>
#include <fstream>
#include "mnist.h"
#include "math.h"
#include "more-math.h"

int reverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;

  ch1 = (i >>  0)&255;
  ch2 = (i >>  8)&255;
  ch3 = (i >> 16)&255;
  ch4 = (i >> 24)&255;

  return ((int) ch1 << 24) +
         ((int) ch2 << 16) +
         ((int) ch3 <<  8) +
         ((int) ch4 <<  0);
}

void loadMnistImages(float**& images, int& numImages) {
  int magicNumber;
  int numRows;
  int numCols;
  numImages = -1;

  std::ifstream file("./mnist/train-images.idx3-ubyte", std::ios::binary);
  if (!file.is_open()) return;

  file.read((char*) &magicNumber , 4);
  file.read((char*) &numImages   , 4);
  file.read((char*) &numRows     , 4);
  file.read((char*) &numCols     , 4);

  magicNumber = reverseInt(magicNumber);
  numImages   = reverseInt(numImages);
  numRows     = reverseInt(numRows);
  numCols     = reverseInt(numCols);

  // Allocate space
  images = new float*[numImages];
  for (int i = 0; i < numImages; i++) {
    int size = pow(2, ceil(log2(numRows*numCols)));
    images[i] = new float[size];
  }

  // Load data
  for (int i = 0; i < numImages; i++) {
    for (int j = 0; j < numRows*numCols; j++) {
      unsigned char pixel = 0;
      file.read((char*) &pixel, 1);

      images[i][j] = (float) pixel;
    }
  }

  // Normalise data
  for (int i = 0; i < numImages; i++)
    for (int j = 0; j < numRows*numCols; j++)
      images[i][j] /= 255.0;
}

void loadMnistLabels(float*& labels, int& numLabels, int digit) {
  int magicNumber;
  numLabels = -1;

  std::ifstream file("./mnist/train-labels.idx1-ubyte", std::ios::binary);
  if (!file.is_open()) return;

  file.read((char*) &magicNumber , 4);
  file.read((char*) &numLabels   , 4);

  magicNumber = reverseInt(magicNumber);
  numLabels   = reverseInt(numLabels);

  // Allocate space
  labels = new float[numLabels];

  // Load data
  for (int i = 0; i < numLabels; i++) {
    char label;
    file.read(&label, 1);

    if ((int) label == digit) {
      labels[i] = 1.0;
    } else {
      labels[i] = 0.0;
    }
  }
}

void loadMnist(float**& images, float*& labels, int& numExamples, int digit) {
  int numImages, numLabels;

  loadMnistImages(images, numImages);
  loadMnistLabels(labels, numLabels, digit);

  if (numImages == numLabels) numExamples = numImages;
  else                        numExamples = -1;
}
