double sm_square(double x, double y){
    return (std::cos(x) + std::cos(y) + 2.0) / 4.0;
}

void init_pot_line(CPUArray& arr, const Float3 l, const Float3 h, const int power, const double sigma, const double amp){
    Float* ptr = arr.ptr_real();
    int3 shape = arr.real_vext();
    assert(shape.z == 1 and shape.y > 1 and shape.x > 1 and "array is not 2d");
    const double f = 2 * M_PI / sigma;
    for(int i1 = 0; i1 < shape.y; i1++){
        const double y = i1 * h.y - l.y / 2.0;
        for(int i0 = 0; i0 < shape.x; i0++){
            const double x = i0 * h.x - l.x / 2.0;
            const double part1 = 0.5 * (tanh(y + sigma / 2) + 1);
            const double part2 = 0.5 * (tanh(- y + sigma / 2) + 1);
            ptr[i1 * shape.x + i0] = amp * std::pow(sm_square(f * x, f * y), power) * part1 * part2;
        }
    }
    std::cout << "\"init_pot_line\" potential initialized" << std::endl;
}
