main: main.cc mnist.cc more-math.cc
	g++ -Wall -std=c++11 -g -O3 -o main main.cc mnist.cc more-math.cc

test: more-math.cc unit.cc blerp.cc tests/*.cc
	g++ -Wall -std=c++11 -g -o test more-math.cc unit.cc blerp.cc tests/*.cc
