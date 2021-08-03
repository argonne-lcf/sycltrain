#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
  sycl::platform P{sycl::gpu_selector{}};
  sycl::device D = P.get_devices(sycl::info::device_type::gpu)[0];
  sycl::context C(D);
  sycl::queue Q(C,D);
  const int N{1729};
  float *A = sycl::malloc_device<float>(N,D,C);
}
