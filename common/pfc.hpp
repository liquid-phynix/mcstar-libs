// XXX: chemical potential of liquid
double mu(double eps, double psi0){
  return  (1.-eps)*psi0+pow(psi0, 3); }
double omega(double eps, double psi0){
  return  0.5*(1.-eps)*pow(psi0, 2) + 0.25*pow(psi0, 4) - mu(eps, psi0) * psi0; }
double omega(double eps, double psi0, double fen){
  return  fen - mu(eps, psi0) * psi0; }

//double calc_free_energy(CPUArray& arr_oper, CPUArray& arr_psi, double eps, double volume){
  //double result = 0;
  //int2 idx;
  //for(idx.x = 0; idx.x < arr_oper.ext<0>(); idx.x++){
    //for(idx.y = 0; idx.y < arr_oper.ext<1>(); idx.y++){
        //const double psi = arr_psi[idx];
        //result += 0.5 * psi * arr_oper[idx] + 0.25 * psi * psi * psi * psi; }}
  //return volume * result / vprod(arr_oper.vext()); }
