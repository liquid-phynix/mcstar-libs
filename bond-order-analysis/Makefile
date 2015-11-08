include ../hostspecific.in

all: libbondorder.so libbondordergsl.so standalone standalone-nogsl

libbondorder.so: bond-order.cpp
	$(CXX) -fPIC --shared -std=c++11 -fopenmp -I$(MCSTARLIBS) bond-order.cpp -o $@

libbondordergsl.so: bond-order-gsl.cpp
	$(CXX) -fPIC --shared -std=c++11 -fopenmp -lgsl -I$(MCSTARLIBS) bond-order-gsl.cpp -o $@

standalone: standalone.cpp
	$(CXX) -std=c++11 standalone.cpp libbondordergsl.so -I$(MCSTARLIBS) -L/usr/lib64/atlas -lgsl -lcblas -o $@

standalone-nogsl: standalone.cpp
	$(CXX) -std=c++11 standalone.cpp libbondorder.so -I$(MCSTARLIBS) -o $@
