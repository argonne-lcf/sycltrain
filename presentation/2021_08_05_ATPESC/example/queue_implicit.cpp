#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
 sycl::queue Q{sycl::gpu_selector{} };
 sycl::device D = Q.get_device();
 sycl::context C = Q.get_context();
}
