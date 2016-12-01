main: main.cc mnist.cc more-math.cc
	g++ -Wall -std=c++11 -g -O0 -ffast-math -o main main.cc mnist.cc more-math.cc
