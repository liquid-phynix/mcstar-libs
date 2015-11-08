// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// XXX: TCLAP custom arguments
struct Int3Arg {
private:
  int3 m_int3;
public:
  Int3Arg(): m_int3(){}
  Int3Arg(int a, int b, int c): m_int3({a, b, c}){}
  Int3Arg& operator=(const std::string& str){
    std::istringstream iss(str); char x1, x2;
    bool ret = iss >> m_int3.x >> x1 >> m_int3.y >> x2 >> m_int3.z;
    if(not ret or not (x1 == 'x' || x1 == 'X') or not (x2 == 'x' || x2 == 'X'))
      throw TCLAP::ArgParseException(str + " is not of form %dx%dx%d");
    return *this; }
  operator int3(){ return m_int3; }
};
struct Int2Arg {
private:
  int2 m_int2;
public:
  Int2Arg(): m_int2(){}
  Int2Arg(int a, int b): m_int2({a, b}){}
  Int2Arg& operator=(const std::string& str){
    std::istringstream iss(str); char x;
    bool ret = iss >> m_int2.x >> x >> m_int2.y;
    if(not ret or not (x == 'x' or x == 'X'))
      throw TCLAP::ArgParseException(str + " is not of form %dx%d");
    return *this; }
  operator int2(){ return m_int2; }
};
namespace TCLAP { template<> struct ArgTraits<Int3Arg> { typedef StringLike ValueCategory; }; }
namespace TCLAP { template<> struct ArgTraits<Int2Arg> { typedef StringLike ValueCategory; }; }
