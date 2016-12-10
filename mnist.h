#ifndef MNIST_H
#define MNIST_H

#include "math.h"
#include "more-math.h"
#include <fstream>
#include <iostream>
#include <random>
#include <string>

float* makeIdentityMap(int n);
float* makeRandomMap(int n);
void makeQuasiConvMapRec(int x, int y, int lenx, int leny, float* map, int &i);
float* makeQuasiConvMap(int n);
void reorderVector(float* v, float* map, int n);
int reverseInt(int i);
void loadMnistImages(float**& images, int& numImages, std::string prefix);
void loadMnistLabels(float*& labels, int& numLabels, int digit, std::string prefix);
void loadMnist(float**& images, float*& labels, int& numExamples, int digit);
void loadMnist(float**& images, float*& labels, int& numExamples, int digit, std::string prefix);
void loadMnist(float**& images, float*& labels, int& numExamples, int digit, std::string prefix, float *map);

#endif
