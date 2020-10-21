#include <CL/sycl.hpp>
#include <stdio.h>
 
int main() {
  int resSycl = 1234;
  {
    cl::sycl::buffer<int, 1> resBuffer(&resSycl, cl::sycl::range<1>(1));
    cl::sycl::queue().submit([&](cl::sycl::handler &cgh) {
        auto resAcc = resBuffer.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.single_task<class X>([=]() {  resAcc[0] = 1; });
      });
  }
 
  int resOmp = 4321;
#pragma omp target map(from: resOmp)
  resOmp = 2;
 
  printf("resSycl = %d\n", resSycl);
  printf("resOmp = %d\n", resOmp);
  return 0;
}
