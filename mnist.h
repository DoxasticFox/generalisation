#ifndef MNIST_H
#define MNIST_H

int reverseInt(int i);
void loadMnistImages(float**& images, int& numImages);
void loadMnistLabels(float*& labels, int& numLabels, int digit);
void loadMnist(float**& images, float*& labels, int& numExamples, int digit);

#endif
