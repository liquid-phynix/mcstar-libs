#pragma once

#include <cstdio>

class TimeAcc {
    cudaEvent_t m_starte, m_ende;
    int m_times;
    float m_elapsed_ms;
    public:
    TimeAcc(): m_times(1), m_elapsed_ms(){
        cudaEventCreate(&m_starte);
        cudaEventCreate(&m_ende); }
    void start(){
        cudaEventRecord(m_starte); }
    void stop(bool incr=true){
        cudaEventRecord(m_ende);
        cudaEventSynchronize(m_ende);
        float elapsed=0;
        cudaEventElapsedTime(&elapsed, m_starte, m_ende);
        if(incr) m_times++;
        m_elapsed_ms += elapsed; }
    void reset(){ m_times = 1; m_elapsed_ms = 0; }
    friend std::ostream& operator<<(std::ostream&, const TimeAcc&);
};
std::ostream& operator<<(std::ostream& o, const TimeAcc& ta){ o << ta.m_elapsed_ms; return o; }
