include ../hostspecific.in

.PHONY: all

all: libcontourline.so main

libcontourline.so: contourline.cpp contourline.hpp
	$(CXX) -fPIC --shared --std=c++11 -I$(MCSTARLIBS) $(IVTK) $(LVTK)  -lvtkFiltersExtraction$(SVTK) contourline.cpp -o $@

main: libcontourline.so main.cpp
	$(CXX) -std=c++11 $(LIBSTDCPP) libcontourline.so main.cpp -o $@
