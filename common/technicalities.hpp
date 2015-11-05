#pragma once

std::ostream& operator<<(std::ostream& o, const dim3&    v){ o << v.x << ' ' << v.y << ' ' << v.z; return o; }
std::ostream& operator<<(std::ostream& o, const float3& v){ o << v.x << ' ' << v.y << ' ' << v.z; return o; }
std::ostream& operator<<(std::ostream& o, const double3& v){ o << v.x << ' ' << v.y << ' ' << v.z; return o; }
std::ostream& operator<<(std::ostream& o, const int3&    v){ o << v.x << ' ' << v.y << ' ' << v.z; return o; }
std::ostream& operator<<(std::ostream& o, const float2& v){ o << v.x << ' ' << v.y; return o; }
std::ostream& operator<<(std::ostream& o, const double2& v){ o << v.x << ' ' << v.y; return o; }
std::ostream& operator<<(std::ostream& o, const int2&    v){ o << v.x << ' ' << v.y; return o; }

bool operator==(const int3&  a, const int3&  b){ return a.x == b.x and a.y == b.y and a.z == b.z; }
bool operator==(const uint3& a, const uint3& b){ return a.x == b.x and a.y == b.y and a.z == b.z; }
bool operator==(const int2&  a, const int2&  b){ return a.x == b.x and a.y == b.y; }
bool operator==(const uint2& a, const uint2& b){ return a.x == b.x and a.y == b.y; }

int3 shape_tr(int3 in){
  int3 out{1, 1, 1};
  int dims = 3;
  if(in.z == 1) dims = 2;
  if(in.y == 1) dims = 1;
  switch(dims){
    case 1:
      out.x = in.x; break;
    case 2:
      out.x = in.y;
      out.y = in.x; break;
    case 3:
      out.x = in.z;
      out.y = in.y;
      out.z = in.x; break;
    default: throw std::runtime_error("shouldn't happen"); }
  return out; }

/*
template <typename Flt> std::ostream& operator<<(std::ostream& o, const std::vector<Flt>& v){
  if(v.size() == 0) return o;
  o << std::scientific;
  o << v[0];
  for(uint i = 1; i < v.size(); i++) o << ' ' << v[i];
  o.precision(5);
  return o; }


int prod(int  v){ return v; }
int prod(int2 v){ return v.x * v.y; }
int prod(int3 v){ return v.x * v.y * v.z; }
int prod(dim3 v){ return v.x * v.y * v.z; }

int3 toint3(const int  v){ return {v, 1, 1}; }
int3 toint3(const int2 v){ return {v.x, v.y, 1}; }
int3 toint3(const int3 v){ return v; }

template <typename Ret> Ret  stdv2vec      (const std::vector<int>&);
template <>             int  stdv2vec<int> (const std::vector<int>& v){ return {v[0]}; }
template <>             int2 stdv2vec<int2>(const std::vector<int>& v){ return {v[0], v[1]}; }
template <>             int3 stdv2vec<int3>(const std::vector<int>& v){ return {v[0], v[1], v[2]}; }

void append_log(const char* fn, const char* str, bool truncate = false){
  FILE* fp = fopen(fn, truncate ? "w" : "a");
  if(fp == NULL){
    fprintf(stderr, "log cannot be appended");
    //std::cerr << "log cannot be appended" << std::endl;
    return; }
  fprintf(fp, "%s", str);
  fclose(fp); }
*/
