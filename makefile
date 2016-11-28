main: main.cc mnist.cc more-math.cc
	g++ -Wall -std=c++11 -g -Ofast -ffast-math -o main main.cc mnist.cc more-math.cc
