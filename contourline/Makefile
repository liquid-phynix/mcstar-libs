include ../hostspecific.in

.PHONY: all

all: libcontourline.so main

libcontourline.so: contourline.cpp contourline.hpp
	g++ -fPIC --shared --std=c++11 -I$(MCSTARLIBS) -lvtkFiltersGeneral -lvtkFiltersCore -lvtkCommonCore -lvtkCommonMisc -lvtkCommonDataModel -lvtkCommonExecutionModel -lvtkCommonSystem -lvtkFiltersExtraction contourline.cpp -o $@

main: libcontourline.so main.cpp
	g++ -std=c++11 libcontourline.so main.cpp -o $@
