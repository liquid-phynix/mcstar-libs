#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <complex>
#include <functional>
#include <omp.h>
#include <common/maximum_struct.hpp>

double ths = -1e6;
double boxsize = 256;
double rlim = 10;
const double dc = 0.7;
const double Pi=M_PI;

const int WI[][3] = {{-4, 0, 4}, {-4, 1, 3}, {-4, 2, 2}, {-4, 3, 1}, {-4, 4, 0}, {-3, -1, 4}, {-3, 0, 3}, {-3, 1, 2}, {-3, 2, 1}, {-3, 3, 0}, {-3, 4, -1}, {-2, -2, 4}, {-2, -1, 3}, {-2, 0, 2}, {-2, 1, 1}, {-2, 2, 0}, {-2, 3, -1}, {-2, 4, -2}, {-1, -3, 4}, {-1, -2, 3}, {-1, -1, 2}, {-1, 0, 1}, {-1, 1, 0}, {-1, 2, -1}, {-1, 3, -2}, {-1, 4, -3}, {0, -4, 4}, {0, -3, 3}, {0, -2, 2}, {0, -1, 1}, {0, 0, 0}, {0, 1, -1}, {0, 2, -2}, {0, 3, -3}, {0, 4, -4}, {1, -4, 3}, {1, -3, 2}, {1, -2, 1}, {1, -1, 0}, {1, 0, -1}, {1, 1, -2}, {1, 2, -3}, {1, 3, -4}, {2, -4, 2}, {2, -3, 1}, {2, -2, 0}, {2, -1, -1}, {2, 0, -2}, {2, 1, -3}, {2, 2, -4}, {3, -4, 1}, {3, -3, 0}, {3, -2, -1}, {3, -1, -2}, {3, 0, -3}, {3, 1, -4}, {4, -4, 0}, {4, -3, -1}, {4, -2, -2}, {4, -1, -3}, {4, 0, -4}};
const double WS[] = {0.10429770312912397,-0.16490914830605122,0.18698939800169145,-0.16490914830605122,0.10429770312912397,-0.16490914830605122,0.15644655469368596,-0.06232979933389715,-0.06232979933389715,0.15644655469368596,-0.16490914830605122,0.18698939800169145,-0.06232979933389715,-0.08194819531574027,0.1413506985480439,-0.08194819531574027,-0.06232979933389715,0.18698939800169145,-0.16490914830605122,-0.06232979933389715,0.1413506985480439,-0.06704852344015112,-0.06704852344015112,0.1413506985480439,-0.06232979933389715,-0.16490914830605122,0.10429770312912397,0.15644655469368596,-0.08194819531574027,-0.06704852344015112,0.13409704688030225,-0.06704852344015112,-0.08194819531574027,0.15644655469368596,0.10429770312912397,-0.16490914830605122,-0.06232979933389715,0.1413506985480439,-0.06704852344015112,-0.06704852344015112,0.1413506985480439,-0.06232979933389715,-0.16490914830605122,0.18698939800169145,-0.06232979933389715,-0.08194819531574027,0.1413506985480439,-0.08194819531574027,-0.06232979933389715,0.18698939800169145,-0.16490914830605122,0.15644655469368596,-0.06232979933389715,-0.06232979933389715,0.15644655469368596,-0.16490914830605122,0.10429770312912397,-0.16490914830605122,0.18698939800169145,-0.16490914830605122,0.10429770312912397};
const int WN = sizeof(WS)/sizeof(double);

struct Triple{ double x, y, z; };
struct Pair{ double a, b; };

typedef std::complex<double> Comp;

struct Point:public Triple{
    Point(double _x, double _y, double _z){ x = _x; y = _y; z = _z; }
    std::list<int> nbor_i;
    std::list<Pair> nbor_vars;
    Comp q4m[9];
    Comp q6m[13];
    Comp q8m[17];
    double q4,q6,q8,w4,q4b,q6b,q8b,w4b;
    int xi;
    double max_norm, larger_norm;
    int max_ind, larger_ind;
};

typedef std::vector<Point> Pvec;

/*
inline double get_diff(double x1, double x2, double w){
  return x1-x2;
  //  double d1=x1-x2;
  //  double d2=d1-w;
  // double d3=d1+w;
  // if(fabs(d1)<fabs(d3)) return d1;
  // else return d3;
}
*/


inline double get_diff(double x1, double x2, double w){
    double d1=x1-x2;
    double d2=d1-w;
    double d3=d1+w;
    if(fabs(d1)<fabs(d2)){
        if(fabs(d1)<fabs(d3)) return d1;
        else return d3;
    }else{
        if(fabs(d2)<fabs(d3)) return d2;
        else return d3;
    }
}


// OMP
void find_nbors(Pvec& pvec, double cutoff, double w){
    int max=pvec.size();

#pragma omp parallel for schedule(dynamic, 2)
    for(int i=0; i<max; i++){
        Point& pi=pvec[i];

        Triple diff; //, c1, c2;
        Pair p;
        double norm, max_norm=0, larger_norm=1e6;
        int max_ind=-1, larger_ind=-1;

        for(int j=0; j<max; j++){
            if(i==j) continue;
            Point& pj=pvec[j];

            // c1.x = pj.x-pi.x; c1.y = pj.y-pi.y; c1.z = pj.z-pi.z;
            // c2.x=c1.x-w; c2.y=c1.y-w; c2.z=c1.z-w;

            diff.x=get_diff(pj.x, pi.x, w); //fabs(c1.x)<fabs(c2.x)?c1.x:c2.x;
            diff.y=get_diff(pj.y, pi.y, w); //fabs(c1.y)<fabs(c2.y)?c1.y:c2.y;
            diff.z=get_diff(pj.z, pi.z, w); //fabs(c1.z)<fabs(c2.z)?c1.z:c2.z;

            norm=sqrt(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);

            if(norm<=cutoff){
                if(norm>max_norm){ max_norm=norm; max_ind=j; }

                p.a=diff.z/norm; p.b=atan2(diff.y, diff.x);

                pi.nbor_i.push_back(j);
                pi.nbor_vars.push_back(p);
            }else if(norm < larger_norm){
                larger_norm = norm;
                larger_ind = j;
            }
        }

        pi.max_norm = max_norm;
        pi.max_ind = max_ind;
        pi.larger_norm = larger_norm;
        pi.larger_ind = larger_ind;
    }
}

template <int n> constexpr double dpow(double x){ return x*dpow<n-1>(x); }
template <> constexpr double dpow<1>(double x){ return x; }

template <int n> constexpr int ipow(int x){ return x*ipow<n-1>(x); }
template <> constexpr int ipow<1>(int x){ return x; }

template <int n> constexpr long int fact(){ return n*fact<n-1>(); }
template <> constexpr long int fact<1>(){ return 1; }
template <> constexpr long int fact<0>(){ return 1; }

inline double prefF(double num, double ath, int m){
    if(m<0){
        if(fabs(1+ath)<1e-3) return 0;
        else return pow((1+ath)/(1-ath), m) / num;
    }else if(m>0){
        if(fabs(1-ath)<1e-3) return 0;
        else return pow((1+ath)/(1-ath), m) / num;
    }else return 1/num;
    if(fabs(1-ath)<1e-3) return 0;
    else return pow((1+ath)/(1-ath), m) / num;
}

template<int m> struct Y4{
    static Comp y4(double ath, double phi){
        Comp c(cos(m * phi), sin(m * phi));
        //    double pref = pow((1+ath)/(1-ath), m) / (Pi * fact<4-m>() * fact<4+m>());
        double pref = prefF(Pi * fact<4-m>() * fact<4+m>(), ath, m);
        pref = 1.5 * sqrt(pref);
        double poly = 9 + 105 * dpow<4>(ath) - 105 * dpow<3>(ath) * m - 10 * ipow<2>(m) + ipow<4>(m) + 5 * ath * m * (11 - 2 * ipow<2>(m)) + 45 * dpow<2>(ath) * (ipow<2>(m) - 2);
        c *= poly; c *= pref;
        return c;
        //    return c!=c ? Comp(0,0) : c;
    }
};
std::function<Comp (double, double)> y4m[9] = { Y4<-4>::y4,Y4<-3>::y4,Y4<-2>::y4,Y4<-1>::y4,Y4<0>::y4,Y4<1>::y4,Y4<2>::y4,Y4<3>::y4,Y4<4>::y4 };

template<int m> struct Y6{
    static Comp y6(double ath, double phi){
        Comp c(cos(m * phi), sin(m * phi));
        //    double pref = 13 * pow((1+ath)/(1-ath), m) / (Pi * fact<6-m>() * fact<6+m>());
        double pref = 13 * prefF(Pi * fact<6-m>() * fact<6+m>(), ath, m);
        pref = 0.5 * sqrt(pref);
        double poly = -225 + 10395 * dpow<6>(ath) - 10395 * dpow<5>(ath) * m + 259 * ipow<2>(m) - 35 * ipow<4>(m) + ipow<6>(m) + 4725 * dpow<4>(ath) * (ipow<2>(m) - 3) - 630 * dpow<3>(ath) * m * (2 * ipow<2>(m) - 17) - 21 * ath * m * (ipow<4>(m) - 25 * ipow<2>(m) + 99) + 105 * dpow<2>(ath) * (2 * ipow<4>(m) - 32 * ipow<2>(m) + 45);
        c *= poly; c *= pref;
        //    return c!=c ? Comp(0,0) : c;
        return c;
    }
};
std::function<Comp (double, double)> y6m[13] = { Y6<-6>::y6,Y6<-5>::y6,Y6<-4>::y6,Y6<-3>::y6,Y6<-2>::y6,Y6<-1>::y6,Y6<0>::y6,Y6<1>::y6,Y6<2>::y6,Y6<3>::y6,Y6<4>::y6,Y6<5>::y6,Y6<6>::y6 };

template<int m> struct Y8{
    static Comp y8(double ath, double phi){
        Comp c(cos(m * phi), sin(m * phi));
        //    double pref = 17 * pow((1+ath)/(1-ath), m) / (Pi * fact<8-m>() * fact<8+m>());
        double pref = 17 * prefF(Pi * fact<8-m>() * fact<8+m>(), ath, m);
        pref = 0.5 * sqrt(pref);
        double poly = 11025 - 396900*dpow<2>(ath) + 2182950*dpow<4>(ath) - 3783780*dpow<6>(ath) + 2027025*dpow<8>(ath) + 136431*ath*m - 1327095*dpow<3>(ath)*m + 3108105*dpow<5>(ath)*m - 2027025*dpow<7>(ath)*m - 12916*ipow<2>(m) + 328545*dpow<2>(ath)*ipow<2>(m) - 1143450*dpow<4>(ath)*ipow<2>(m) + 945945*dpow<6>(ath)*ipow<2>(m) - 39564*ath*ipow<3>(m) + 242550*dpow<3>(ath)*ipow<3>(m) - 270270*dpow<5>(ath)*ipow<3>(m) + 1974*ipow<4>(m) - 31500*dpow<2>(ath)*ipow<4>(m) + 51975*dpow<4>(ath)*ipow<4>(m) + 2394*ath*ipow<5>(m) - 6930*dpow<3>(ath)*ipow<5>(m) - 84*ipow<6>(m) + 630*dpow<2>(ath)*ipow<6>(m) - 36*ath*ipow<7>(m) + ipow<8>(m);
        c *= poly; c *= pref;
        //    return c!=c ? Comp(0,0) : c;
        return c;
    }
};
std::function<Comp (double, double)> y8m[17] = { Y8<-8>::y8, Y8<-7>::y8, Y8<-6>::y8,Y8<-5>::y8,Y8<-4>::y8,Y8<-3>::y8,Y8<-2>::y8,Y8<-1>::y8,Y8<0>::y8,Y8<1>::y8,Y8<2>::y8,Y8<3>::y8,Y8<4>::y8,Y8<5>::y8,Y8<6>::y8,Y8<7>::y8,Y8<8>::y8 };

double q4(Point& atom){
    double sum=0;
    for(int m = -4; m <= 4; m++) sum += pow(fabs(atom.q4m[m+4]),2);
    return sqrt(4*Pi/9 * sum);
}
double q6(Point& atom){
    double sum=0;
    for(int m = -6; m <= 6; m++) sum += pow(fabs(atom.q6m[m+6]),2);
    return sqrt(4*Pi/13 * sum);
}
double q8(Point& atom){
    double sum=0;
    for(int m = -8; m <= 8; m++) sum += pow(fabs(atom.q8m[m+8]),2);
    return sqrt(4*Pi/17 * sum);
}

double w4(Point& atom){
    Comp csum(0,0);
    for(int i=0; i<WN; i++){
        Comp ctmp=WS[i];
        ctmp *= atom.q4m[WI[i][0] + 4]; ctmp *= atom.q4m[WI[i][1] + 4]; ctmp *= atom.q4m[WI[i][2] + 4];
        csum += ctmp;
    }
    double sum = 0;
    for(int m = -4; m <= 4; m++)  sum += pow(fabs(atom.q4m[m+4]),2);
    return csum.real()/sqrt(dpow<3>(sum));
}

double w4b(Point& atom, Pvec& pvec){
    Comp q4mbar[9]; for(int i=0; i<9; i++) q4mbar[i]=atom.q4m[i];
    for(int nbi: atom.nbor_i) for(int i=0; i<9; i++) q4mbar[i]=pvec[nbi].q4m[i];
    for(int i=0; i<9; i++) q4mbar[i]/=(double)atom.nbor_i.size();

    Comp csum(0,0);
    for(int i=0; i<WN; i++){
        Comp ctmp=WS[i];
        ctmp *= q4mbar[WI[i][0] + 4]; ctmp *= q4mbar[WI[i][1] + 4]; ctmp *= q4mbar[WI[i][2] + 4];
        csum += ctmp;
    }
    double sum = 0;
    for(int m = -4; m <= 4; m++)  sum+=pow(fabs(q4mbar[m+4]),2);
    return csum.real()/sqrt(dpow<3>(sum));
}

double q4b(Point& atom, Pvec& pvec){
    double sum=0;
    for(int m = -4; m <= 4; m++){
        Comp csum=atom.q4m[m+4];
        for(int nbi: atom.nbor_i) csum += pvec[nbi].q4m[m+4];
        csum /= 1 + (double)atom.nbor_i.size();
        sum += pow(fabs(csum),2);
    }
    return sqrt(4*Pi/9 * sum);
}
double q6b(Point& atom, Pvec& pvec){
    double sum=0;
    for(int m = -6; m <= 6; m++){
        Comp csum=atom.q6m[m+6];
        for(int nbi: atom.nbor_i) csum += pvec[nbi].q6m[m+6];
        csum /= 1 + (double)atom.nbor_i.size();
        sum += pow(fabs(csum),2);
    }
    return sqrt(4*Pi/13 * sum);
}
double q8b(Point& atom, Pvec& pvec){
    double sum=0;
    for(int m = -8; m <= 8; m++){
        Comp csum=atom.q8m[m+8];
        for(int nbi: atom.nbor_i) csum += pvec[nbi].q8m[m+8];
        csum /= 1 + (double)atom.nbor_i.size();
        sum += pow(fabs(csum),2);
    }
    return sqrt(4*Pi/17 * sum);
}

void q_lm(Point& atom){
    for(int m=-4; m<=4; m++) atom.q4m[m+4]=Comp(0,0);
    for(int m=-6; m<=6; m++) atom.q6m[m+6]=Comp(0,0);
    for(int m=-8; m<=8; m++) atom.q8m[m+8]=Comp(0,0);
    for(Pair& p: atom.nbor_vars){
        for(int m=-4; m<=4; m++) atom.q4m[m+4] += y4m[m+4](p.a, p.b);
        for(int m=-6; m<=6; m++) atom.q6m[m+6] += y6m[m+6](p.a, p.b);
        for(int m=-8; m<=8; m++) atom.q8m[m+8] += y8m[m+8](p.a, p.b);
    }
    double N=atom.nbor_vars.size();
    for(int m=-4; m<=4; m++) atom.q4m[m+4]/=N;
    for(int m=-6; m<=6; m++) atom.q6m[m+6]/=N;
    for(int m=-8; m<=8; m++) atom.q8m[m+8]/=N;
}

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
            that_denom += pow(fabs(pvec[nbi].q6m[i]), 2);
        }
        that_denom = sqrt(that_denom);

        double crit = d6.real() / (this_denom * that_denom);
        if(crit > 0.7) s++;
    }
    return s;
}

template <typename T>
void bond_order_analysis(Maximum<T>* input, int input_len, double h0, double h1, double h2, double _ths, double _boxsize, double _rlim, FILE* fp){
    ths = _ths; boxsize = _boxsize; rlim = _rlim;
    // input: a vector of N*4 elements, {x0, y0, z0, psi0, x1, x2, ...}
    Pvec pvec;
    for(int i = 0; i < input_len; i++){
        Maximum<T>& m = input[i];
        if(m.field > ths)
            pvec.push_back(Point(m.i0 * h0, m.i1 * h1, m.i2 * h2));
    }
    std::cerr << pvec.size() << " atoms above threshold" << std::endl;
    find_nbors(pvec, rlim, boxsize);
    std::cerr << "all nbors computed" << std::endl;

    // q_lm(i)
#pragma omp parallel for //schedule(dynamic, 1)
    for(int i=0; i<pvec.size(); i++) if(pvec[i].nbor_vars.size() > 0) q_lm(pvec[i]);

    // q_l(i), q_l_bar(i)
#pragma omp parallel for //schedule(dynamic, 1)
    for(int i=0; i<pvec.size(); i++){
        Point& atom=pvec[i];
        if(atom.nbor_vars.size() == 0) continue;
        atom.q4 = q4(atom);
        atom.q6 = q6(atom);
        atom.q8 = q8(atom);
        atom.w4 = w4(atom);
        atom.q4b = q4b(atom, pvec);
        atom.q6b = q6b(atom, pvec);
        atom.q8b = q8b(atom, pvec);
        atom.w4b = w4b(atom, pvec);
    }
#pragma omp parallel for //schedule(dynamic, 1)
    for(int i=0; i<pvec.size(); i++){
        Point& atom=pvec[i];
        if(atom.nbor_vars.size() == 0) continue;
        atom.xi = xi(atom, pvec);
    }

    //FILE* outfile = fopen(outfn.c_str(), "w");
    //if(outfile == NULL){
        //std::cerr << "output file " << outfile << " cannot be created" << std::endl;
        //exit(EXIT_FAILURE);
    //}
    fprintf(fp, "# x y z #nb xi q4 q6 q8 w4 q4b q6b q8b w4b\n");
    for(Point& atom : pvec){
        if(atom.nbor_vars.size() == 0) continue;
        //fprintf(fp, "%.10e %.10e %.10e %d %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
        fprintf(fp, "%.6f\t%.6f\t%.6f\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                atom.x, atom.y, atom.z, atom.nbor_vars.size(), atom.xi, atom.q4, atom.q6, atom.q8, atom.w4, atom.q4b, atom.q6b, atom.q8b, atom.w4b);
    }
    fflush(fp);
    //fclose(outfile);
    /*
       for(Point& atom : pvec){
       if(atom.nbor_vars.size() == 0) continue;
       printf("%d\n",atom.nbor_vars.size());
       }
       cerr << "#nb" << endl;
       */
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
