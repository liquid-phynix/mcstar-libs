#include <vector>
#include <iostream>
#include <cstdlib>
#include "maximum_struct.hpp"
#include "bond-order.hpp"

typedef Maximum<double> M;

int main(int argc, char* argv[]){
    if(argc != 3){
        std::cerr << "usage: " << argv[0] << " <edgelen> <rlim>" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<M> maxima;
    std::cin.ignore(1024, '\n');
    std::cin.ignore(1024, '\n');
    while(not std::cin.eof()){
        //if(maxima.size()%100==0) std::cerr << "vector length=" << maxima.size() << std::endl;
        M m;
        m.field = 1;
        while(true){
            int c = std::cin.get();
            if(c == ' ' or c == '\t') break;
        }
        std::cin >> m.i0 >> std::ws >> m.i1 >> std::ws >> m.i2 >> std::ws;
        maxima.push_back(m);
    }
    //std::cerr << "vector length=" << maxima.size() << std::endl;
    // bond_order_analysis(          , input_len    , h0, h1, h2, ths , boxsize      , rlim         , fp);
    bond_order_analysis(maxima.data(), maxima.size(), 1 , 1 , 1 , -1e6, atof(argv[1]), atof(argv[2]), stdout);

    return 0;
}
