main: main.cc mnist.cc more-math.cc
	g++ -Wall -std=c++11 -O3 -ffast-math -o main main.cc mnist.cc more-math.cc
