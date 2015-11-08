#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <ctgmath>
#include <complex>
#include <functional>
#include <utility>
#include <omp.h>
#include <gsl/gsl_sf_legendre.h>
#include <common/maximum_struct.hpp>

double ths = -1e6;
double boxsize = 256;
double rlim = 10;
const double dc = 0.7;
const double Pi=M_PI;

const int WI[][3] = {{-4, 0, 4}, {-4, 1, 3}, {-4, 2, 2}, {-4, 3, 1}, {-4, 4, 0}, {-3, -1, 4}, {-3, 0, 3}, {-3, 1, 2}, {-3, 2, 1}, {-3, 3, 0}, {-3, 4, -1}, {-2, -2, 4}, {-2, -1, 3}, {-2, 0, 2}, {-2, 1, 1}, {-2, 2, 0}, {-2, 3, -1}, {-2, 4, -2}, {-1, -3, 4}, {-1, -2, 3}, {-1, -1, 2}, {-1, 0, 1}, {-1, 1, 0}, {-1, 2, -1}, {-1, 3, -2}, {-1, 4, -3}, {0, -4, 4}, {0, -3, 3}, {0, -2, 2}, {0, -1, 1}, {0, 0, 0}, {0, 1, -1}, {0, 2, -2}, {0, 3, -3}, {0, 4, -4}, {1, -4, 3}, {1, -3, 2}, {1, -2, 1}, {1, -1, 0}, {1, 0, -1}, {1, 1, -2}, {1, 2, -3}, {1, 3, -4}, {2, -4, 2}, {2, -3, 1}, {2, -2, 0}, {2, -1, -1}, {2, 0, -2}, {2, 1, -3}, {2, 2, -4}, {3, -4, 1}, {3, -3, 0}, {3, -2, -1}, {3, -1, -2}, {3, 0, -3}, {3, 1, -4}, {4, -4, 0}, {4, -3, -1}, {4, -2, -2}, {4, -1, -3}, {4, 0, -4}};
const double WS[] = {0.10429770312912397,-0.16490914830605122,0.18698939800169145,-0.16490914830605122,0.10429770312912397,-0.16490914830605122,0.15644655469368596,-0.06232979933389715,-0.06232979933389715,0.15644655469368596,-0.16490914830605122,0.18698939800169145,-0.06232979933389715,-0.08194819531574027,0.1413506985480439,-0.08194819531574027,-0.06232979933389715,0.18698939800169145,-0.16490914830605122,-0.06232979933389715,0.1413506985480439,-0.06704852344015112,-0.06704852344015112,0.1413506985480439,-0.06232979933389715,-0.16490914830605122,0.10429770312912397,0.15644655469368596,-0.08194819531574027,-0.06704852344015112,0.13409704688030225,-0.06704852344015112,-0.08194819531574027,0.15644655469368596,0.10429770312912397,-0.16490914830605122,-0.06232979933389715,0.1413506985480439,-0.06704852344015112,-0.06704852344015112,0.1413506985480439,-0.06232979933389715,-0.16490914830605122,0.18698939800169145,-0.06232979933389715,-0.08194819531574027,0.1413506985480439,-0.08194819531574027,-0.06232979933389715,0.18698939800169145,-0.16490914830605122,0.15644655469368596,-0.06232979933389715,-0.06232979933389715,0.15644655469368596,-0.16490914830605122,0.10429770312912397,-0.16490914830605122,0.18698939800169145,-0.16490914830605122,0.10429770312912397};
const int WN = sizeof(WS)/sizeof(WS[0]);

struct Triple{ double x, y, z; };

struct Pair{ double a, b; };

typedef std::complex<double> Comp;

struct Point : public Triple {
    Point(double _x, double _y, double _z){
        x = _x; y = _y; z = _z;
        for(int i = 0; i < sizeof(q4m) / sizeof(Comp); i++) q4m[i] = {};
        for(int i = 0; i < sizeof(q6m) / sizeof(Comp); i++) q6m[i] = {};
        for(int i = 0; i < sizeof(q8m) / sizeof(Comp); i++) q8m[i] = {};
        q4 = 0; q6 = 0; q8 = 0; w4 = 0; q4b = 0; q6b = 0; q8b = 0; w4b = 0;
        max_norm = 0; larger_norm = 0; max_ind = 0; larger_ind = 0;
    }
    std::list<int> nbor_i;
    std::list<double> cos_theta;
    std::list<Comp> exp_i_phi;
    Comp q4m[9];
    Comp q6m[13];
    Comp q8m[17];
    double q4,q6,q8,w4,q4b,q6b,q8b,w4b;
    int xi;
    double max_norm, larger_norm;
    int max_ind, larger_ind;
};

typedef std::vector<Point> Pvec;

inline double image_diff(double x2, double x1, double w){
    double dx = x2 - x1;
    if(std::fabs(dx) > w/2){
        if(dx > 0) return dx - w;
        else return dx + w;
    } else return dx;
}

template <typename F> F ipow(F x, int e){
    if(e < 0) return F(1) / ipow(x, -e);
    if(e == 0) return 1;
    else if(e == 1) return x;
    else if(e % 2 == 0) { F tmp = ipow(x, e / 2); return tmp * tmp; }
    else return x * ipow(x, e - 1);
}

// OMP
void find_nbors(Pvec& pvec, const double cutoff, const double w){
    const int max=pvec.size();

#pragma omp parallel for schedule(dynamic, 2)
    for(int i=0; i<max; i++){
        Point& pi=pvec[i];

        double max_norm=0, larger_norm=1e6;
        int max_ind=-1, larger_ind=-1;

        for(int j=0; j<max; j++){
            if(i==j) continue;
            Point& pj=pvec[j];

            Triple diff {image_diff(pj.x, pi.x, w), image_diff(pj.y, pi.y, w), image_diff(pj.z, pi.z, w)};
            double rp = diff.x*diff.x+diff.y*diff.y;
            const double r = sqrt(rp + diff.z*diff.z);
            rp = sqrt(rp);

            if(r < cutoff){
                if(r > max_norm){ max_norm=r; max_ind=j; }
                // p.a === cos(theta) = z / r
                pi.cos_theta.push_back(diff.z / r);
                // p.b === cos(phi) + i * sin(phi)
                if(rp < 1e-12)
                    // phi == pi / 2
                    pi.exp_i_phi.push_back(Comp(0, 1));
                else
                    pi.exp_i_phi.push_back(Comp(diff.x / rp, diff.y / rp));

                pi.nbor_i.push_back(j);
            }else if(r < larger_norm){
                larger_norm = r;
                larger_ind = j;
            }
        }
        pi.max_norm = max_norm;
        pi.max_ind = max_ind;
        pi.larger_norm = larger_norm;
        pi.larger_ind = larger_ind;
    }
}

double w4(Point& atom){
    const int l = 4;
    const int len = 2 * l + 1;
    Comp csum(0,0);
    for(int i=0; i<WN; i++){
        Comp ctmp=WS[i];
        for(int ii = 0; ii < 3; ii++){
            ctmp *= atom.q4m[WI[i][ii] + 4];
        }
        csum += ctmp;
    }
    return csum.real() / std::pow(atom.q4, 3) / std::pow(len / 4 / Pi, 1.5);
}

//double w4b(Point& atom, Pvec& pvec){
    //Comp q4mbar[9]; for(int i=0; i<9; i++) q4mbar[i]=atom.q4m[i];
    //for(int nbi: atom.nbor_i) for(int i=0; i<9; i++) q4mbar[i]=pvec[nbi].q4m[i];
    //for(int i=0; i<9; i++) q4mbar[i]/=(double)atom.nbor_i.size();

    //Comp csum(0,0);
    //for(int i=0; i<WN; i++){
        //Comp ctmp=WS[i];
        //ctmp *= q4mbar[WI[i][0] + 4]; ctmp *= q4mbar[WI[i][1] + 4]; ctmp *= q4mbar[WI[i][2] + 4];
        //csum += ctmp;
    //}
    //double sum = 0;
    //for(int m = -4; m <= 4; m++)  sum+=pow(fabs(q4mbar[m+4]),2);
    //return csum.real()/sqrt(dpow<3>(sum));
//}

double ql(Comp* arr, int l){
    const int len = 2 * l + 1;
    double sum=0;
    for(int i = 0; i < len; i++){
        double tmp = std::fabs(arr[i]);
        sum += tmp * tmp; }
    return sqrt(4*Pi/len * sum);
}

double q4b(Point& atom, Pvec& pvec){
    const int l = 4;
    const int len = 2 * l + 1;
    double sum=0;
    for(int i = 0; i < len; i++){
        Comp csum=atom.q4m[i];
        for(int nbi: atom.nbor_i) csum += pvec[nbi].q4m[i];
        csum /= atom.nbor_i.size() + 1;
        double tmp = std::fabs(csum);
        sum += tmp * tmp;
    }
    return sqrt(4*Pi/len * sum);
}
double q6b(Point& atom, Pvec& pvec){
    const int l = 6;
    const int len = 2 * l + 1;
    double sum=0;
    for(int i = 0; i < len; i++){
        Comp csum=atom.q6m[i];
        for(int nbi: atom.nbor_i) csum += pvec[nbi].q6m[i];
        csum /= atom.nbor_i.size() + 1;
        double tmp = std::fabs(csum);
        sum += tmp * tmp;
    }
    return sqrt(4*Pi/len * sum);
}
double q8b(Point& atom, Pvec& pvec){
    const int l = 8;
    const int len = 2 * l + 1;
    double sum=0;
    for(int i = 0; i < len; i++){
        Comp csum=atom.q8m[i];
        for(int nbi: atom.nbor_i) csum += pvec[nbi].q8m[i];
        csum /= atom.nbor_i.size() + 1;
        double tmp = std::fabs(csum);
        sum += tmp * tmp;
    }
    return sqrt(4*Pi/len * sum);
}

void q_lm(Point& atom){
    std::list<double>::iterator it1;
    std::list<Comp>::iterator it2;

    for(it1 = atom.cos_theta.begin(), it2 = atom.exp_i_phi.begin(); it1 != atom.cos_theta.end(); it1++, it2++){
        for(int m = 0; m <= 8; m++){
            Comp exp_i_phi_to_m = ipow(*it2, m);
            Comp sign = m % 2 ? -1 : 1;

            Comp tmp = gsl_sf_legendre_sphPlm(8, m, *it1) * exp_i_phi_to_m;
            atom.q8m[m+8] += tmp;
            if(m != 0) atom.q8m[-m+8] += std::conj(tmp) * sign;

            if(m>6) continue;
            tmp = gsl_sf_legendre_sphPlm(6, m, *it1) * exp_i_phi_to_m;
            atom.q6m[m+6] += tmp;
            if(m != 0) atom.q6m[-m+6] += std::conj(tmp) * sign;

            if(m>4) continue;
            tmp = gsl_sf_legendre_sphPlm(4, m, *it1) * exp_i_phi_to_m;
            atom.q4m[m+4] += tmp;
            if(m != 0) atom.q4m[-m+4] += std::conj(tmp) * sign;
        }
    }
    double Nb=atom.nbor_i.size();
    for(int m=0; m<9; m++)  atom.q4m[m] /= Nb;
    for(int m=0; m<13; m++) atom.q6m[m] /= Nb;
    for(int m=0; m<17; m++) atom.q8m[m] /= Nb;
}

/*
int xi(Point& atom, Pvec& pvec){
    double this_denom = 0;
    for(int i=0; i<13; i++) this_denom += pow(fabs(atom.q6m[i]), 2);
    this_denom = sqrt(this_denom);

    int s=0;
    for(int nbi: atom.nbor_i){
        double that_denom = 0;
        Comp d6(0, 0);
        for(int i=0; i<13; i++){
            Comp other(pvec[nbi].q6m[i].real(), -pvec[nbi].q6m[i].imag());
            d6 += atom.q6m[i] * other;
            that_denom += pow(std::fabs(pvec[nbi].q6m[i]), 2);
        }
        that_denom = sqrt(that_denom);

        double crit = d6.real() / (this_denom * that_denom);
        if(crit > 0.7) s++;
    }
    return s;
}
*/

template <typename T>
void bond_order_analysis(Maximum<T>* input, int input_len, double h0, double h1, double h2, double _ths, double _boxsize, double _rlim, FILE* fp){
    ths = _ths; boxsize = _boxsize; rlim = _rlim;

    Pvec pvec;
    for(int i = 0; i < input_len; i++){
        Maximum<T>& m = input[i];
        if(m.field > ths)
            pvec.push_back(Point(m.i0 * h0, m.i1 * h1, m.i2 * h2));
    }
    std::cerr << "bond order > " << pvec.size() << " atoms above threshold" << std::endl;
    find_nbors(pvec, rlim, boxsize);
    std::cerr << "bond order > all nbors computed" << std::endl;

    // q_lm(i)
#pragma omp parallel for //schedule(dynamic, 1)
    for(int i=0; i<pvec.size(); i++) if(pvec[i].nbor_i.size() > 0) q_lm(pvec[i]);

    // q_l(i), q_l_bar(i)
#pragma omp parallel for //schedule(dynamic, 1)
    for(int i=0; i<pvec.size(); i++){
        Point& atom=pvec[i];
        if(atom.nbor_i.size() == 0) continue;
        atom.q4 = ql(atom.q4m, 4);
        atom.q6 = ql(atom.q6m, 6);
        atom.q8 = ql(atom.q8m, 8);
        //atom.w4 = w4(atom);
        atom.q4b = q4b(atom, pvec);
        atom.q6b = q6b(atom, pvec);
        atom.q8b = q8b(atom, pvec);
        //atom.w4b = w4b(atom, pvec);
    }
#pragma omp parallel for //schedule(dynamic, 1)
    for(int i=0; i<pvec.size(); i++){
        Point& atom=pvec[i];
        if(atom.nbor_i.size() == 0) continue;
        //atom.xi = xi(atom, pvec);
    }

    //FILE* outfile = fopen(outfn.c_str(), "w");
    //if(outfile == NULL){
        //std::cerr << "output file " << outfile << " cannot be created" << std::endl;
        //exit(EXIT_FAILURE);
    //}
    //fprintf(fp, "# x y z #nb xi q4 q6 q8 w4 q4b q6b q8b w4b\n");
    fprintf(fp, "# x y z #nb q4 q6 q8 q4b q6b q8b\n");
    for(Point& atom : pvec){
        if(atom.nbor_i.size() == 0) continue;
        // xi
        //fprintf(fp, "%.6f\t%.6f\t%.6f\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                //atom.x, atom.y, atom.z, atom.nbor_i.size(), atom.xi, atom.q4, atom.q6, atom.q8, atom.w4, atom.q4b, atom.q6b, atom.q8b, atom.w4b);
        // no xi
        fprintf(fp, "%.6f\t%.6f\t%.6f\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                atom.x, atom.y, atom.z, atom.nbor_i.size(), atom.q4, atom.q6, atom.q8, atom.q4b, atom.q6b, atom.q8b);
    }
    fflush(fp);
}

template void bond_order_analysis<float>(Maximum<float>* input, int input_len, double h0, double h1, double h2, double _ths, double _boxsize, double _rlim, FILE* fp);
template void bond_order_analysis<double>(Maximum<double>* input, int input_len, double h0, double h1, double h2, double _ths, double _boxsize, double _rlim, FILE* fp);
template void bond_order_analysis<int>(Maximum<int>* input, int input_len, double h0, double h1, double h2, double _ths, double _boxsize, double _rlim, FILE* fp);
// g++ -O3 -std=c++11 struct.cpp -fopenmp -o struct.cpp.run
// env OMP_THREAD_LIMIT=1

// q_lm(i)     -- Y_lm atlaga (i) szomszedjaira
// q_l(i)      -- sqrt(4Pi/(2l+1) * sum(abs(q_lm(i))^2, m=-l..l))
// q_lm_bar(i) -- q_lm(k) atlaga (i)-re es (i) szomszedjaira
// q_l_bar(i)  -- sqrt(4Pi/(2l+1) * sum(abs(q_lm_bar(i))^2, m=-l..l))
