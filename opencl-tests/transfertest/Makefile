CXX=g++
ifeq ($(shell hostname -s),guenhwyvar)
CXX=/opt/rh/devtoolset-2/root/usr/bin/g++
endif

all: test_normal test_ahp test_uhp test_uhp_map
test_normal: test_normal.cpp
	$(CXX) -O3 -std=c++11 -I/usr/local/amd/include/ -lOpenCL -o $@ test_normal.cpp -Wno-deprecated-declarations
test_ahp: test_ahp.cpp
	$(CXX) -O3 -std=c++11 -I/usr/local/amd/include/ -lOpenCL -o $@ test_ahp.cpp -Wno-deprecated-declarations
test_uhp: test_uhp.cpp
	$(CXX) -O3 -std=c++11 -I/usr/local/amd/include/ -lOpenCL -o $@ test_uhp.cpp -Wno-deprecated-declarations
test_uhp_map: test_uhp_map.cpp
	$(CXX) -O3 -std=c++11 -I/usr/local/amd/include/ -lOpenCL -o $@ test_uhp_map.cpp -Wno-deprecated-declarations
