#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
 sycl::queue Q{sycl::gpu_selector{} };
 const int N{1729};
 float *A = sycl::malloc_device<float>(N,Q);
}
