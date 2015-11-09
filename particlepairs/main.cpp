#include <ctgmath>
#include <vector>
#include <iostream>
#include <cstdlib>

struct Triple { double x, y, z; };

inline double image_diff(double x2, double x1, double w){
    double dx = x2 - x1;
    if(std::fabs(dx) > w/2){
        if(dx > 0) return dx - w;
        else return dx + w;
    } else return dx;
}

int main(int argc, char* argv[]){
    if(argc != 2){
        std::cerr << "usage: " << argv[0] << " <edgelen>" << std::endl;
        exit(EXIT_FAILURE);
    }
    double xd, yd, zd;
    double box = atof(argv[1]);

    std::vector<Triple> points;
    std::cin.ignore(1024, '\n');
    std::cin.ignore(1024, '\n');
    while(not std::cin.eof()){
        //if(maxima.size()%100==0) std::cerr << "vector length=" << maxima.size() << std::endl;
        Triple t;
        while(true){
            int c = std::cin.get();
            if(c == ' ' or c == '\t') break;
        }
        std::cin >> t.x >> std::ws >> t.y >> std::ws >> t.z >> std::ws;
        points.push_back(t);
    }
    std::cerr << "# of particles = " << points.size() << std::endl;
    // bond_order_analysis(          , input_len    , h0, h1, h2, ths , boxsize      , rlim         , fp);

    int maxi = points.size();
    for(int i = 0; i < maxi; i++){
        Triple& p1 = points[i];
        for(int j = 0; j < maxi; j++){
            Triple& p2 = points[j];
            if(i == j) continue;
            xd = image_diff(p2.x, p1.x, box);
            yd = image_diff(p2.y, p1.y, box);
            zd = image_diff(p2.z, p1.z, box);
            std::cout << std::sqrt(xd * xd + yd * yd + zd * zd) << "\n";
        }
    }

    return 0;
}
// g++ -std=c++11 main.cpp -o main
