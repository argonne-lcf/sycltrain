#include <sycl/sycl.hpp>
#include "tprint.hpp"

int main() {
  //         __                     ___
  //  /\    (_  o ._ _  ._  |  _     |  _.  _ |
  // /--\   __) | | | | |_) | (/_    | (_| _> |<
  //                    |

  // A "queue" is bound to a "device", and is used to submid work ("commands")
  // to the associated device.
  sycl::queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  // [=]: Capture the outside scope (variables) by value
  // (): Declaring an anonymouns function (lambda) without parameters
  // {...}: Body of the anonymouns function
  auto f = [=]() {
    syclx::printf("Hello, World for lambda!\n");
  };
  // Submit one work item (a single task) to the GPU using the previous lambda
  // Queue submission are asyncrhonous (similar to OpenMP nowait)
  Q.single_task(f);
  // We wait for all the commands submitted to the queue to complete
  // In this case only one
  Q.wait();
  return 0;
}
