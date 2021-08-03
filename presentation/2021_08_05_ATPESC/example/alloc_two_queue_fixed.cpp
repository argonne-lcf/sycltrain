#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

void f_implicit(const int N){
 sycl::queue Q{sycl::gpu_selector{}};
 float *A = sycl::malloc_device<float>(N,Q);
 float *B = sycl::malloc_device<float>(N,Q);
 Q.memcpy(A,B,N*sizeof(float)).wait();
}

void f_explicit(const int N){
  sycl::platform P{sycl::gpu_selector{}};
  sycl::device D = P.get_devices(sycl::info::device_type::gpu)[0];
  sycl::context C(D);

  sycl::queue Q1(C,D);
  float *A = sycl::malloc_device<float>(N,Q1);
  sycl::queue Q2(C,D);
  float *B = sycl::malloc_device<float>(N,Q2);
  Q1.memcpy(A,B,N*sizeof(float)).wait();
}

int main(){
    const int N{10};
    f_explicit(N);
    f_implicit(N);
}
