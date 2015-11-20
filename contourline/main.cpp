#include <iostream>
#include <cmath>
#include "contourline.hpp"

int main(int, char *[]){
    int n0 = 200;
    int n1 = 300;
    float h0 = 1. / n0;
    float h1 = 1. / n1;
    float* array = new float[n0 * n1];
    for(int i0 = 0; i0 < n0; i0++){
        for(int i1 = 0; i1 < n1; i1++){
            array[i1 * n0 + i0] = std::sin(3 * i0 * h0) * std::sin(4 * i1 * h1); }}

    Lines lines = contourlines(array, n0, n1, h0, h1, 0.5);
    delete[] array;

    std::cerr << "# of lines " << lines.size() << std::endl;

    Line& line = lines[0];
    for(int i = 0; i < line.size(); i++){
        std::cout << line[i].first << " " << line[i].second << "\n";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
