include ../hostspecific.in

.PHONY: all

all: libcontourline.so main

libcontourline.so: contourline.cpp contourline.hpp
	$(CXX) -fPIC --shared --std=c++11 -I$(MCSTARLIBS) $(IVTK) $(LVTK) -lvtkFiltersGeneral$(SVTK) -lvtkFiltersCore$(SVTK) -lvtkCommonCore$(SVTK) -lvtkCommonMisc$(SVTK) -lvtkCommonDataModel$(SVTK) -lvtkCommonExecutionModel$(SVTK) -lvtkCommonSystem$(SVTK) -lvtkFiltersExtraction$(SVTK) contourline.cpp -o $@

main: libcontourline.so main.cpp
	$(CXX) -std=c++11 libcontourline.so main.cpp -o $@
