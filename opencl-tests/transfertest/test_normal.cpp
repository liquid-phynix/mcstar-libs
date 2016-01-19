#define __CL_ENABLE_EXCEPTIONS
#include "cl-1.2.hpp"
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include <sys/mman.h>
#include "timing.hpp"

void print_platform_info(cl::Platform& p){
  std::string str;
  p.getInfo(CL_PLATFORM_NAME, &str);
  std::cout << "platform: " << str << std::endl;
}

void print_device_info(cl::Device& d){
  std::string str;
  d.getInfo(CL_DEVICE_NAME, &str);
  std::cout << "device: " << str << std::endl;
}

// time differences are in nanoseconds (1e-9)
double start_ms(cl::Event& e){
  cl_ulong start = e.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong end = e.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return (end - start) * 1e-6; }

double submit_ms(cl::Event& e){
  cl_ulong start = e.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
  cl_ulong end = e.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return (end - start) * 1e-6; }
double queued_ms(cl::Event& e){
  cl_ulong start = e.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
  cl_ulong end = e.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  return (end - start) * 1e-6; }

void _main(int pnum, int dnum){
    const int start_exp = 26;
    const cl_ulong LIST_LEN = 1 << start_exp;
    const cl_ulong LIST_SIZE = sizeof(int) * LIST_LEN;
    int* host_ptr = new int[LIST_LEN]{};
    if(mlock(host_ptr, LIST_SIZE) != 0)
        std::cerr << "host memory not pinned" << std::endl;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    //for(auto& p: platforms) print_platform_info(p);
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pnum])(), 0 };
    cl::Context context(CL_DEVICE_TYPE_ALL, cps);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[dnum], CL_QUEUE_PROFILING_ENABLE);

    print_platform_info(platforms[pnum]);
    print_device_info(devices[dnum]);

    std::ifstream sourceFile("kernel.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program(context, source); program.build(devices);
    cl::Kernel increment(program, "increment");

    cl::Buffer buffer(context, CL_MEM_READ_WRITE, LIST_SIZE);
    queue.enqueueFillBuffer<char>(buffer, 0, 0, LIST_SIZE);
    increment.setArg(0, buffer);


    { // warm-up
        const cl_ulong dummy = 124341534;
        increment.setArg(1, dummy);
        increment.setArg(2, dummy);
        queue.enqueueWriteBuffer(buffer, CL_FALSE, 0, LIST_SIZE, host_ptr);
        queue.enqueueNDRangeKernel(increment, cl::NullRange, cl::NDRange(LIST_LEN), cl::NullRange);
        queue.enqueueReadBuffer(buffer, CL_FALSE, 0, LIST_SIZE, host_ptr);
        queue.finish();
    }

    const cl_ulong transfer = 1 * LIST_LEN;
    const int min_exp = 20;
    std::cout << "size\t\ttimes\thost w\t\thost rw\t\tdev r s\t\tdev w s\t\tdev k s\t\tdiff%\n";
    for(int exp = start_exp; exp >= min_exp; exp--){
        const cl_ulong len = 1 << exp;
        const cl_ulong size = sizeof(int) * len;
        const int iters = transfer / len;
        increment.setArg(1, len);

        TimeAcc host_write, host_read_write;
        TimeAcc device_all;
        double device_buffer_read[3]{}; // queued - submit - start
        double device_buffer_write[3]{};
        double device_kernel[3]{};

        for(cl_ulong iter = 0; iter < iters; iter++){
            // initialize on HOST
            host_write.start();
            for(int i = 0; i < len; i++){
                host_ptr[i] = iter + len + i;
            }
            host_write.stop();

            increment.setArg(2, iter);
            // change content on DEVICE
            cl::Event device_r, device_w, device_k;
            device_all.start();
            queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, host_ptr, NULL, &device_r);
            queue.enqueueNDRangeKernel(increment, cl::NullRange, cl::NDRange(len), cl::NullRange, NULL, &device_k);
            queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_ptr, NULL, &device_w);
            queue.finish();
            device_all.stop();

            device_buffer_read[0] += queued_ms(device_r);
            device_buffer_read[1] += submit_ms(device_r);
            device_buffer_read[2] += start_ms(device_r);

            device_buffer_write[0] += queued_ms(device_w);
            device_buffer_write[1] += submit_ms(device_w);
            device_buffer_write[2] += start_ms(device_w);

            device_kernel[0] += queued_ms(device_k);
            device_kernel[1] += submit_ms(device_k);
            device_kernel[2] += start_ms(device_k);

            // read and check data on HOST
            host_read_write.start();
            for(int i = 0; i < len; i++){
                host_ptr[i] = host_ptr[i] == iter;
            }
            host_read_write.stop();

            int sum = 0;
            for(int i = 0; i < len; i++){
                sum += host_ptr[i];
            }
            if(sum != len){
                std::cerr << "sum failure: sum=" << sum << " len=" << len << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        device_buffer_read[0] /= iters; device_buffer_read[1] /= iters; device_buffer_read[2] /= iters;
        device_buffer_write[0] /= iters; device_buffer_write[1] /= iters; device_buffer_write[2] /= iters;
        device_kernel[0] /= iters; device_kernel[1] /= iters; device_kernel[2] /= iters;

        double mb = size / 1024. / 1024.;
        double bw_host_write = mb / host_write.get_avg_ms() * 1e3;
        double bw_host_read_write = 2 * mb / host_read_write.get_avg_ms() * 1e3;
        double bw_dev_read = mb / device_buffer_read[2] * 1e3;
        double bw_dev_write = mb / device_buffer_write[2] * 1e3;
        double bw_dev_kernel = 2 * mb / device_kernel[2] * 1e3;
        double device_all_time = device_buffer_read[2] + device_buffer_write[2] + device_kernel[2];
        double difft_percent = (device_all.get_avg_ms() - device_all_time) / device_all_time * 100;
        printf("%.2e\t%d\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%8.2f\t%.2f\n",
                mb, iters, bw_host_write, bw_host_read_write, bw_dev_read, bw_dev_write, bw_dev_kernel, difft_percent);
    }
    std::cout << "size\t\ttimes\thost w\t\thost rw\t\tdev r s\t\tdev w s\t\tdev k s\t\tdiff%\n";

  delete[] host_ptr;
}

int main(int argc, char* argv[]) {
    int pnum = 0;
    int dnum = 0;
    if(argc >= 3)
        dnum = atoi(argv[2]);
    if(argc >= 2)
        pnum = atoi(argv[1]);

    try { _main(pnum, dnum); }
    catch(cl::Error error) {
        std::cerr << "Exception: " << error.what() << "(" << error.err() << ")" << std::endl; }
    return 0;
}
