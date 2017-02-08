#ifndef MNIST_H
#define MNIST_H

#include "math.h"
#include "more-math.h"
#include <fstream>
#include <iostream>
#include <random>
#include <string>

double* makeIdentityMap(int n);
double* makeRandomMap(int n);
void makeQuasiConvMapRec(int x, int y, int lenx, int leny, double* map, int &i);
double* makeQuasiConvMap(int n);
void reorderVector(double* v, double* map, int n);
int reverseInt(int i);
void loadMnistImages(double**& images, int& numImages, std::string prefix);
void loadMnistLabels(double*& labels, int& numLabels, int digit, std::string prefix);
void loadMnist(double**& images, double*& labels, int& numExamples, int digit);
void loadMnist(double**& images, double*& labels, int& numExamples, int digit, std::string prefix);
void loadMnist(double**& images, double*& labels, int& numExamples, int digit, std::string prefix, double *map);

#endif
