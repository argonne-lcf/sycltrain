#include <CL/sycl.hpp>

// Maybe in the futur sycl will note be in the 'cl' namespace
namespace sycl = cl::sycl;

int main() {
  // Selectors determine which device kernels will be dispatched to.
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  sycl::cpu_selector selector;

  sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";
  //         __                     ___
  //  /\    (_  o ._ _  ._  |  _     |  _.  _ |
  // /--\   __) | | | | |_) | (/_    | (_| _> |<
  //                    |

  // Create a command_group to issue command to the group.
  // Use A lambda to generate the control group handler
  // Queue submision are asyncrhonous! (similar to OpenMP nowait)
  myQueue.submit([&](sycl::handler &cgh) {
    // Create a output stream
    sycl::stream sout(1024, 256, cgh);
    // Submit a unique task, using a lambda
    cgh.single_task<class hello_world>([=]() {
      sout << "Hello, World!" << sycl::endl;
    }); // End of the kernel function
  });   // End of the queue commands. The kernel is now submited

  return 0;
}
