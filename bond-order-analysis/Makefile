INC=/home/mcstar/src/libsrc/
CXX=g++-4.8

all: libbondorder.so libbondordergsl.so standalone standalone-nogsl

libbondorder.so: bond-order.cpp
	$(CXX) -fPIC --shared -std=c++11 -fopenmp -I$(INC) bond-order.cpp -o $@

libbondordergsl.so: bond-order-gsl.cpp
	$(CXX) -fPIC --shared -std=c++11 -fopenmp -lgsl -I$(INC) bond-order-gsl.cpp -o $@

standalone: standalone.cpp
	$(CXX) -std=c++11 standalone.cpp libbondordergsl.so -lgsl -lcblas -o $@

standalone-nogsl: standalone.cpp
	$(CXX) -std=c++11 standalone.cpp libbondorder.so -o $@
