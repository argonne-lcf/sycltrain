#include <CL/sycl.hpp>
#include <cstdint>

// Maybe in the futur sycl will not be in the 'cl' namespace
namespace sycl = cl::sycl;

int main() {
  // Selectors determine which device kernels will be dispatched to.
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  sycl::gpu_selector selector;
  
  sycl::queue Q(selector);
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  //         __                     ___
  //  /\    (_  o ._ _  ._  |  _     |  _.  _ |
  // /--\   __) | | | | |_) | (/_    | (_| _> |<
  //                    |

  // Create a command_group to issue command to the group.
  // Use A lambda to generate the control group handler
  // Queue submision are asyncrhonous (similar to OpenMP nowait)
  Q.submit([&](sycl::handler &cgh) {
    // Create a output stream
    sycl::stream sout(1024, 256, cgh);
    // Submit a unique task, using a lambda
    cgh.single_task([=]() {
      sout << "Hello, World!" << sycl::endl;
    }); // End of the kernel function
  });   // End of the queue commands. The kernel is now submited

  // wait for all queue submissions to complete
  Q.wait();

  return 0;
}
