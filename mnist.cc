#include "mnist.h"

double* makeIdentityMap(int n) {
  double *map = new double[n];

  for (int i = 0; i < n; i++)
    map[i] = i;

  return map;
}

double* makeRandomMap(int n) {
  double *map = makeIdentityMap(n);

  // Shuffle map
  for (int i = n-1; i >= 0; --i){
      int j = rand() % (i+1);

      int temp = map[i];
      map[i] = map[j];
      map[j] = temp;
  }

  return map;
}

void makeQuasiConvMapRec(int x, int y, int lenx, int leny, double* map, int &i) {
  if (lenx == 0 || leny == 0) return;
  if (lenx == 1 && leny == 1) {
    map[i] = 28 * x + y;
    i++;
    return;
  }

  int lenLeft = (lenx+1)/2; int lenRight = lenx - lenLeft;
  int lenTop  = (leny+1)/2; int lenBot   = leny - lenTop;

  int xLeft = x; int xRight = xLeft+lenLeft;
  int yTop  = y; int yBot   = yTop +lenTop;

  makeQuasiConvMapRec(xLeft,  yTop, lenLeft,  lenTop, map, i); // Top left
  makeQuasiConvMapRec(xRight, yTop, lenRight, lenTop, map, i); // Top right
  makeQuasiConvMapRec(xLeft,  yBot, lenLeft,  lenBot, map, i); // Bot left
  makeQuasiConvMapRec(xRight, yBot, lenRight, lenBot, map, i); // Bot right
}

double* makeQuasiConvMap(int n) {
  double *map = makeIdentityMap(n);

  int i = 0;
  makeQuasiConvMapRec(0, 0, 28, 28, map, i);

  return map;
}

void reorderVector(double* v, double* map, int n) {
  double *tmp = new double[n];

  // 1. Put the reordered vector components in tmp
  for (int i = 0; i < n; i++) {
    int idx = map[i];
    tmp[i] = v[idx];
  }

  // 2. Copy tmp's contents to v
  for (int i = 0; i < n; i++)
    v[i] = tmp[i];

  delete[] tmp;
}

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

void loadMnistImages(
    double**& images,
    int& numImages,
    std::string prefix,
    double *map
) {
  int magicNumber;
  int numRows;
  int numCols;
  numImages = -1;

  std::string filename = "./mnist/" + prefix + "-images.idx3-ubyte";
  std::ifstream file(filename, std::ios::binary);
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
  images = new double*[numImages];
  for (int i = 0; i < numImages; i++) {
    int size = pow(2, ceil(log2(numRows*numCols)));
    images[i] = new double[size*2];
  }

  // Load data
  for (int i = 0; i < numImages; i++) {
    for (int j = 0; j < numRows*numCols; j++) {
      unsigned char pixel = 0;
      file.read((char*) &pixel, 1);

      images[i][j     ] = (double) pixel;
      images[i][j+1024] = (double) pixel;
    }
  }

  // Normalise data
  for (int i = 0; i < numImages; i++)
    for (int j = 0; j < 2048; j++)
      images[i][j] /= 255.0;

  // Reorder data
  if (map)
    for (int i = 0; i < numImages; i++) {
      reorderVector( images[i],       map, numRows*numCols);
      reorderVector(&images[i][1024], map, numRows*numCols);
    }

}

void loadMnistLabels(
    double*& labels,
    int& numLabels,
    int digit,
    std::string prefix
) {
  int magicNumber;
  numLabels = -1;

  std::string filename = "./mnist/" + prefix + "-labels.idx1-ubyte";
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) return;

  file.read((char*) &magicNumber , 4);
  file.read((char*) &numLabels   , 4);

  magicNumber = reverseInt(magicNumber);
  numLabels   = reverseInt(numLabels);

  // Allocate space
  labels = new double[numLabels];

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

void loadMnist(
    double**& images,
    double*& labels,
    int& numExamples,
    int digit,
    std::string prefix,
    double *map
) {
  int numImages, numLabels;

  loadMnistImages(images, numImages, prefix, map);
  loadMnistLabels(labels, numLabels, digit, prefix);

  if (numImages == numLabels) numExamples = numImages;
  else                        numExamples = -1;
}

void loadMnist(
    double**& images,
    double*& labels,
    int& numExamples,
    int digit,
    std::string prefix
) {
  loadMnist(images, labels, numExamples, digit, prefix, 0);
}

void loadMnist(
    double**& images,
    double*& labels,
    int& numExamples,
    int digit
) {
  loadMnist(images, labels, numExamples, digit, "train");
}
