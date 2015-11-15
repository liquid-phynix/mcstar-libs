#define PI M_PI
#define PIX2 (2.0 * M_PI)

dim3 i2d3(const int  v){ return dim3(v,   1,   1); }
dim3 i2d3(const int2 v){ return dim3(v.x, v.y, 1); }
dim3 i2d3(const int3 v){ return dim3(v.x, v.y, v.z); }

//typedef boost::rational<int> Irat;

inline int norm2(int3 v){ return v.x * v.x + v.y * v.y + v.z * v.z; }
inline double norm2(double3 v){ return v.x * v.x + v.y * v.y + v.z * v.z; }
inline double norm2(double2 v){ return v.x * v.x + v.y * v.y; }
inline double dist2(double2 v1, double2 v2){ return norm2(double2{v1.x - v2.x, v1.y - v2.y}); }
inline double norm(double3 v){ return sqrt(norm2(v)); }
inline double norm(double2 v){ return sqrt(norm2(v)); }
inline double norm(int3 v){ return sqrt(norm2(v)); }
inline double3 normalized(int3 v){ double n = norm(v); return { v.x / n, v.y / n, v.z / n }; }
inline double3 normalized(double3 v){ double n = norm(v); return { v.x / n, v.y / n, v.z / n }; }
inline double2 normalized(double2 v){ double n = norm(v); return { v.x / n, v.y / n }; }
inline double inner(double3 v1, double3 v2){ return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline double inner(double2 v1, double2 v2){ return v1.x * v2.x + v1.y * v2.y; }
inline double inner(int3 v1, double3 v2){ return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline int3 cross_int3(int3 pv, int3 qv){
  return { pv.y * qv.z - pv.z * qv.y,
           pv.z * qv.x - pv.x * qv.z,
           pv.x * qv.y - pv.y * qv.x }; }

  /*
int3 canonical_miller(int3 m){
  if(m.x == 0 and m.y == 0 and m.z == 0){
    std::cerr << "Miller-indices cannot be zero all at once" << std::endl; exit(-1); }
  if(m.x < 0) m.x = - m.x; if(m.y < 0) m.y = - m.y; if(m.z < 0) m.z = - m.z;
  int vec[3] = {m.x, m.y, m.z}; std::sort(vec, vec + 3);
  int common = boost::gcd(vec[0], boost::gcd(vec[1], vec[2]));
  return { vec[0] / common, vec[1] / common, vec[2] / common}; }

double z_full_mult(int3 pv, int3 qv, double* _d = NULL){
  int3 zv = cross_int3(pv, qv); int denom = norm2(zv);
  Irat cx(zv.x, denom); Irat cy(zv.y, denom); Irat cz(zv.z, denom);
  int mult = boost::lcm(cx.denominator(), boost::lcm(cy.denominator(), cz.denominator()));
  double d = 1.0 / sqrt(denom);
  if(_d) *_d = d;
  std::cout << "crystal plane spacing(sigma=1): " << d << "\n"
            << "z-periodic multiplier(sigma=1): " << mult << std::endl;
  return mult * d; }
  */

double condition_2x2(double* arr){
  double a = arr[0] * arr[0] + arr[2] * arr[2];
  double bc = arr[0] * arr[1] + arr[2] * arr[3];
  double d = arr[1] * arr[1] + arr[3] * arr[3];
  double eig1 = (a + d - sqrt(a * a + 4.0 * bc * bc - 2.0 * a * d + d * d)) / 2.0;
  double eig2 = (a + d + sqrt(a * a + 4.0 * bc * bc - 2.0 * a * d + d * d)) / 2.0;
  assert(eig1 > 0.0 and "eigenvalue must be positive");
  assert(eig2 > 0.0 and "eigenvalue must be positive");
  return sqrt(eig1 > eig2 ? eig1 : eig2); }

double crystal_condition(int3 pv, int3 qv){
  double normP = norm(pv);
  int3 perp = cross_int3(pv, qv);
  int3 newv = cross_int3(pv, perp);
  double3 nvu = normalized(newv);
  double3 pvu = normalized(pv);
  double mat[4] = { normP, inner(qv, pvu),
                    0,     inner(qv, nvu) };
  return condition_2x2(mat); }

  /*
int3 integrify(int i1, int i2, Irat r3){
  return {i1 * r3.denominator(), i2 * r3.denominator(), r3.numerator() * r3.denominator()}; }

double find_basis(int m1, int m2, int m3, int3& pv_out, int3& qv_out){
  int mill2 = m1 * m1 + m2 * m2 + m3 * m3;
  const int w = 20;
  double condition = std::numeric_limits<double>::infinity();
  for(int n1 = -w; n1 <= w; n1++){
    for(int n2 = -w; n2 <= w; n2++){
      for(int n3 = -w; n3 <= w; n3++){
        for(int n4 = -w; n4 <= w; n4++){
          int3 pv = integrify(n1, n2, Irat(- (n1 * m1 + n2 * m2), m3));
          int3 qv = integrify(n3, n4, Irat(- (n3 * m1 + n4 * m2), m3));
          int nnn2 = norm2(cross_int3(pv, qv));
          if(mill2 == nnn2){
            double cond = crystal_condition(pv, qv);
            if(cond < condition){
              condition = cond;
              pv_out = pv;
              qv_out = qv; }}}}}}
  assert(condition != std::numeric_limits<double>::infinity() and "find_basis failed");
  return condition; }
  */

// calculate the max(abs(.)) convergence from the difference of the prev and current timesteps
/*
template <typename Flt> double calc_conv_eq_max(Array::CPUArray<Flt, 2>& ar){
  double conv = fabs(ar[int2({0,0})]);
  ar.over([&conv](Flt v){ conv = conv > fabs(v) ? conv : fabs(v); });
  return conv; }
template <typename Flt> double calc_conv_eq_max(Array::CPUArray<Flt, 3>& ar){
  double conv = fabs(ar[int3({0,0,0})]);
  ar.over([&conv](Flt v){ conv = conv > fabs(v) ? conv : fabs(v); });
  return conv; }
*/

// calculates lattice indices for the atomic position closest to a given point
// in a 2D triangular lattice, returns pair or integers and distance
std::pair<int2,double> calc_closest_lattice_indices(int2 n1n2, double2 tro_a, double2 tro_b, double2 xpyp){
  const double d1 = dist2(double2{n1n2.x * tro_a.x + n1n2.y * tro_b.x,
                                  n1n2.x * tro_a.y + n1n2.y * tro_b.y},         xpyp);
  const double d2 = dist2(double2{(n1n2.x+1) * tro_a.x + n1n2.y * tro_b.x,
                                  (n1n2.x+1) * tro_a.y + n1n2.y * tro_b.y},     xpyp);
  const double d3 = dist2(double2{n1n2.x * tro_a.x + (n1n2.y+1) * tro_b.x,
                                  n1n2.x * tro_a.y + (n1n2.y+1) * tro_b.y},     xpyp);
  const double d4 = dist2(double2{(n1n2.x+1) * tro_a.x + (n1n2.y+1) * tro_b.x,
                                  (n1n2.x+1) * tro_a.y + (n1n2.y+1) * tro_b.y}, xpyp);
  if(d1 < d2 and d1 < d3 and d1 < d4)
    return std::make_pair(n1n2, d1);
  else if(d2 < d3 and d2 < d4)
    return std::make_pair(int2{n1n2.x+1, n1n2.y},d2);
  else if(d3 < d4)
    return std::make_pair(int2{n1n2.x, n1n2.y+1}, d3);
  else
    return std::make_pair(int2{n1n2.x+1, n1n2.y+1}, d4); }

 bool factors235(int n, int i = 2){
     if(i * i > n) return n <= 5;
     else if(n % i == 0) return factors235(i) and factors235(n / i);
     else return factors235(n, i+1);
 }

 int _nextoneup(int n){
   while(true){
     n++;
     if(factors235(n))
       return n;
   }
 }

 int roundup235(int n, int i = 2){
 //    std::cout << "roundup235 " << n << std::endl;
     if(i * i > n){
         if(n <= 5){
           return n;
         } else {
           return _nextoneup(n);
         }
     } else if(n % i == 0) return roundup235(i) * roundup235(n / i);
     else return roundup235(n, i+1);
 }

int isgn(int val){
    return (val > 0) - (val < 0);
}
