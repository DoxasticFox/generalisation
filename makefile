main: main.cc mnist.cc more-math.cc
	g++ -Wall -std=c++11 -g -O2 -o main main.cc mnist.cc more-math.cc
