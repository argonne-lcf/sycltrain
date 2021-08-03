#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
 const int N{1729};

 sycl::queue Q1{sycl::gpu_selector{}};
 float *A = sycl::malloc_device<float>(N,Q1);

 sycl::queue Q2{sycl::gpu_selector{}};
 float *B = sycl::malloc_device<float>(N,Q2);
 
 Q1.memcpy(A,B,N*sizeof(float));
}
