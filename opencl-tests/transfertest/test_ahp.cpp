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

    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, LIST_SIZE);
    queue.enqueueFillBuffer<char>(buffer, 0, 0, LIST_SIZE);
    increment.setArg(0, buffer);


    { // warm-up
        const cl_ulong dummy = 124341534;
        increment.setArg(1, dummy);
        increment.setArg(2, dummy);
        int* host_ptr = (int*)queue.enqueueMapBuffer(buffer, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, LIST_SIZE);
        queue.enqueueNDRangeKernel(increment, cl::NullRange, cl::NDRange(LIST_LEN), cl::NullRange);
        queue.enqueueUnmapMemObject(buffer, host_ptr);
        queue.finish();
    }

    const cl_ulong transfer = 1 * LIST_LEN;
    const int min_exp = 10;
    std::cout << "size\t\ttimes\thost w\t\thost rw\t\tmap1\t\tunmap1\t\tmap2\t\tunmap2\t\tk bw\t\tdiff%\n";
    for(int exp = start_exp; exp >= min_exp; exp--){
        const cl_ulong len = 1 << exp;
        const cl_ulong size = sizeof(int) * len;
        const int iters = transfer / len;
        increment.setArg(1, len);

        TimeAcc host_write, host_read_write;
        TimeAcc device_all;
        double map_1_buffer_time[3]{}; // queued - submit - start
        double map_2_buffer_time[3]{};
        double unmap_1_buffer_time[3]{};
        double unmap_2_buffer_time[3]{};
        double device_kernel[3]{};

        for(cl_ulong iter = 0; iter < iters; iter++){
            increment.setArg(2, iter);
            cl::Event map_1_buffer, unmap_1_buffer, device_k;
            cl::Event map_2_buffer, unmap_2_buffer;
            // initialize on HOST
            // CL_MAP_READ : @map device -> host; @unmap -
            // CL_MAP_WRITE : @map device -> host; @unmap host -> device
            // CL_MAP_WRITE_INVALIDATE_REGION : @map -; @unmap host -> device
            int* host_ptr = (int*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, LIST_SIZE, NULL, &map_1_buffer);
            host_write.start();
            for(int i = 0; i < len; i++){
                host_ptr[i] = iter + len + i;
            }
            host_write.stop();
            // CL_MAP_READ | CL_MAP_WRITE => host_ptr -> buffer
            queue.enqueueUnmapMemObject(buffer, host_ptr, NULL, &unmap_1_buffer);

            // change content on DEVICE
            device_all.start();
            queue.enqueueNDRangeKernel(increment, cl::NullRange, cl::NDRange(len), cl::NullRange, NULL, &device_k);
            queue.finish();
            device_all.stop();


            // read and check data on HOST
            host_ptr = (int*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ, 0, LIST_SIZE, NULL, &map_2_buffer);
            host_read_write.start();
            for(int i = 0; i < len; i++){
                host_ptr[i] = host_ptr[i] == iter;
            }
            host_read_write.stop();
            queue.enqueueUnmapMemObject(buffer, host_ptr, NULL, &unmap_2_buffer);
            queue.finish();



            device_k.wait();
            device_kernel[0] += queued_ms(device_k);
            device_kernel[1] += submit_ms(device_k);
            device_kernel[2] += start_ms (device_k);
            map_1_buffer.wait();
            unmap_1_buffer.wait();
            map_1_buffer_time[0] += queued_ms(map_1_buffer);
            map_1_buffer_time[1] += submit_ms(map_1_buffer);
            map_1_buffer_time[2] += start_ms (map_1_buffer);
            unmap_1_buffer_time[0] += queued_ms(unmap_1_buffer);
            unmap_1_buffer_time[1] += submit_ms(unmap_1_buffer);
            unmap_1_buffer_time[2] += start_ms (unmap_1_buffer);
            map_2_buffer.wait();
            unmap_2_buffer.wait();
            map_2_buffer_time[0] += queued_ms(map_2_buffer);
            map_2_buffer_time[1] += submit_ms(map_2_buffer);
            map_2_buffer_time[2] += start_ms (map_2_buffer);
            unmap_2_buffer_time[0] += queued_ms(unmap_2_buffer);
            unmap_2_buffer_time[1] += submit_ms(unmap_2_buffer);
            unmap_2_buffer_time[2] += start_ms (unmap_2_buffer);

            int sum = 0;
            for(int i = 0; i < len; i++){
                sum += host_ptr[i];
            }
            if(sum != len){
                std::cerr << "sum failure: sum=" << sum << " len=" << len << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        map_1_buffer_time[0] /= iters;
        map_1_buffer_time[1] /= iters;
        map_1_buffer_time[2] /= iters;
        map_2_buffer_time[0] /= iters;
        map_2_buffer_time[1] /= iters;
        map_2_buffer_time[2] /= iters;
        unmap_1_buffer_time[0] /= iters;
        unmap_1_buffer_time[1] /= iters;
        unmap_1_buffer_time[2] /= iters;
        unmap_2_buffer_time[0] /= iters;
        unmap_2_buffer_time[1] /= iters;
        unmap_2_buffer_time[2] /= iters;
        device_kernel[0] /= iters;
        device_kernel[1] /= iters;
        device_kernel[2] /= iters;

        double mb = size / 1024. / 1024.;
        double bw_host_write = mb / host_write.get_avg_ms() * 1e3;
        double bw_host_read_write = 2 * mb / host_read_write.get_avg_ms() * 1e3;
        double bw_dev_map1 = mb / map_1_buffer_time[2] * 1e3;
        double bw_dev_map2 = mb / map_2_buffer_time[2] * 1e3;
        double bw_dev_unmap1 = mb / unmap_1_buffer_time[2] * 1e3;
        double bw_dev_unmap2 = mb / unmap_2_buffer_time[2] * 1e3;
        double bw_dev_kernel = 2 * mb / device_kernel[2] * 1e3;
        double device_all_time = device_kernel[2];
        double difft_percent = (device_all.get_avg_ms() - device_all_time) / device_all_time * 100;
        printf("%.2e\t%d\t%8.2f\t%8.2f"
               "\t%8.2f\t%8.2f\t%8.2f\t%8.2f"
               "\t%8.2f\t%.2f\n",
               mb, iters, bw_host_write, bw_host_read_write,
               bw_dev_map1, bw_dev_unmap1, bw_dev_map2, bw_dev_unmap2,
               bw_dev_kernel, difft_percent);
    }
    std::cout << "size\t\ttimes\thost w\t\thost rw\t\tmap1\t\tunmap1\t\tmap2\t\tunmap2\t\tk bw\t\tdiff%\n";

}

int main(int argc, char* argv[]) {
    int pnum = 0;
    int dnum = 0;
    if(argc >= 3)
        dnum = atoi(argv[2]);
    if(argc >= 2)
        pnum = atoi(argv[1]);

#define ECODE(code) case code: std::cerr<<#code<<std::endl;break;
    try { _main(pnum, dnum); }
    catch(cl::Error error) {
        std::cerr << "Exception: " << error.what() << "(" << error.err() << ")" << std::endl;
        switch(error.err()){
            ECODE(CL_DEVICE_NOT_FOUND )
            ECODE(CL_DEVICE_NOT_AVAILABLE )
            ECODE(CL_COMPILER_NOT_AVAILABLE )
            ECODE(CL_MEM_OBJECT_ALLOCATION_FAILURE )
            ECODE(CL_OUT_OF_RESOURCES )
            ECODE(CL_OUT_OF_HOST_MEMORY )
            ECODE(CL_PROFILING_INFO_NOT_AVAILABLE )
            ECODE(CL_MEM_COPY_OVERLAP )
            ECODE(CL_IMAGE_FORMAT_MISMATCH )
            ECODE(CL_IMAGE_FORMAT_NOT_SUPPORTED )
            ECODE(CL_BUILD_PROGRAM_FAILURE )
            ECODE(CL_MAP_FAILURE )
            ECODE(CL_MISALIGNED_SUB_BUFFER_OFFSET )
            ECODE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST )
            ECODE(CL_COMPILE_PROGRAM_FAILURE )
            ECODE(CL_LINKER_NOT_AVAILABLE )
            ECODE(CL_LINK_PROGRAM_FAILURE )
            ECODE(CL_DEVICE_PARTITION_FAILED )
            ECODE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE )
            ECODE(CL_INVALID_DEVICE_TYPE )
            ECODE(CL_INVALID_PLATFORM )
            ECODE(CL_INVALID_DEVICE )
            ECODE(CL_INVALID_CONTEXT )
            ECODE(CL_INVALID_QUEUE_PROPERTIES )
            ECODE(CL_INVALID_COMMAND_QUEUE )
            ECODE(CL_INVALID_HOST_PTR )
            ECODE(CL_INVALID_MEM_OBJECT )
            ECODE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR )
            ECODE(CL_INVALID_IMAGE_SIZE )
            ECODE(CL_INVALID_SAMPLER )
            ECODE(CL_INVALID_BINARY )
            ECODE(CL_INVALID_BUILD_OPTIONS )
            ECODE(CL_INVALID_PROGRAM )
            ECODE(CL_INVALID_PROGRAM_EXECUTABLE )
            ECODE(CL_INVALID_KERNEL_NAME )
            ECODE(CL_INVALID_KERNEL_DEFINITION )
            ECODE(CL_INVALID_KERNEL )
            ECODE(CL_INVALID_ARG_INDEX )
            ECODE(CL_INVALID_ARG_VALUE )
            ECODE(CL_INVALID_ARG_SIZE )
            ECODE(CL_INVALID_KERNEL_ARGS )
            ECODE(CL_INVALID_WORK_DIMENSION )
            ECODE(CL_INVALID_WORK_GROUP_SIZE )
            ECODE(CL_INVALID_WORK_ITEM_SIZE )
            ECODE(CL_INVALID_GLOBAL_OFFSET )
            ECODE(CL_INVALID_EVENT_WAIT_LIST )
            ECODE(CL_INVALID_EVENT )
            ECODE(CL_INVALID_OPERATION )
            ECODE(CL_INVALID_GL_OBJECT )
            ECODE(CL_INVALID_BUFFER_SIZE )
            ECODE(CL_INVALID_MIP_LEVEL )
            ECODE(CL_INVALID_GLOBAL_WORK_SIZE )
            ECODE(CL_INVALID_PROPERTY )
            ECODE(CL_INVALID_IMAGE_DESCRIPTOR )
            ECODE(CL_INVALID_COMPILER_OPTIONS )
            ECODE(CL_INVALID_LINKER_OPTIONS )
            ECODE(CL_INVALID_DEVICE_PARTITION_COUNT )
            default:
                std::cerr << "error code not handled" << std::endl;
        }
    }
    return 0;
}

