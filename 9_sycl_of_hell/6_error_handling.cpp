#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {
  const auto global_range = (size_t)atoi(argv[1]);
  const auto local_range = (size_t)atoi(argv[2]);

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`

  // Create you async handler
  sycl::async_handler ah = [](sycl::exception_list elist) {
    for (auto e : elist) {
      std::rethrow_exception(e);
    }
  };
  sycl::queue myQueue(selector, ah);
  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // ._   _|        ._ _. ._   _   _
  // | | (_|        | (_| | | (_| (/_
  //           __              _|

  try {
    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create a output stream (lot of display, lot of number)
      sycl::stream cout(10240, 2560, cgh);

      // nd_range, geneate a nd_item who allow use to query loop dispach
      // information
      cgh.parallel_for<class hello_world>(
          sycl::nd_range<1>{sycl::range<1>(global_range),
                            sycl::range<1>(local_range)},
          [=](sycl::nd_item<1> idx) {
            const int world_rank = idx.get_global_id(0);
            const int work_size = idx.get_global_range(0);
            cout << "Hello world: World rank/size: " << world_rank << " / "
                 << work_size << sycl::endl;
          }); // End of the kernel function
    });       // End of the queue commands

    myQueue.wait_and_throw();
  } catch (sycl::exception &e) {
    std::cout << "Async Exception: " << e.what() << std::endl;
  }

  return 0;
}
