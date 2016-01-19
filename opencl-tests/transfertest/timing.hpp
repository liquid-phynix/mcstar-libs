#include <sys/time.h>
#include <cstdio>

class TimeAcc {
  int m_times;
  long long int m_elapsed; // in usec (1e-6s)
  double m_elapsed_avg;
  timeval m_tval;
public:
  TimeAcc(): m_times(0), m_elapsed(), m_elapsed_avg(), m_tval(){}
  void start(){ gettimeofday(&m_tval, NULL); }
  void stop(bool incr=true){
    timeval end;
    gettimeofday(&end, NULL);
    if(incr) m_times++;
    m_elapsed += (end.tv_sec - m_tval.tv_sec) * 1000 * 1000 + (end.tv_usec - m_tval.tv_usec);
    m_elapsed_avg = double(m_elapsed) / m_times; }
  void reset(){ m_times = 0; m_elapsed = 0; m_elapsed_avg = 0; }
  friend std::ostream& operator<<(std::ostream&, const TimeAcc&);
  double get_ms(){ return m_elapsed * 1e-3; }
  double get_avg_ms(){
      return m_elapsed / double(m_times) * 1e-3;
  }
};
std::ostream& operator<<(std::ostream& o, const TimeAcc& ta){ o << (ta.m_elapsed_avg * 1e-3); return o; }

