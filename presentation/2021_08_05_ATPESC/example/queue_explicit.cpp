#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
  sycl::platform P{sycl::gpu_selector{}};
  sycl::device D = P.get_devices(sycl::info::device_type::gpu)[0];
  sycl::context C(D);
  sycl::queue Q(C,D);
}
