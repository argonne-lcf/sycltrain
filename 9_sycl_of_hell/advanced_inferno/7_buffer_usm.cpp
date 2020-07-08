#include "cxxopts.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  cxxopts::Options options("7 buffer usm",
                           " How to use Unifed shared memory");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"));

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const auto global_range = result["grange"].as<int>();

 //            _           __                                              
 // | | ._  o _|_ _   _|   (_  |_   _. ._ _   _|   ._ _   _  ._ _   _  ._   
 // |_| | | |  | (/_ (_|   __) | | (_| | (/_ (_|   | | | (/_ | | | (_) | \/ 
 //                                                                      / 

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  sycl::queue myQueue(selector);

  sycl::device dev = myQueue.get_device();
  sycl::context ctex = myQueue.get_context();

  int *A = static_cast<int *>(sycl::malloc_shared(global_range * sizeof(int), dev, ctex));

  // Advise runtime how memory will be used
  //auto e = myQueue.mem_advise(A, global_range * sizeof(int), PI_MEM_ADVICE_SET_NON_ATOMIC_MOSTLY);
  //e.wait();

  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // Create a command_group to issue command to the group
  myQueue.submit([&](sycl::handler &cgh) {
    // No accessor needed!
    cgh.parallel_for<class hello_world>(
        sycl::range<1>{sycl::range<1>(global_range)},
        [=](sycl::item<1> id) {
          const sycl::id<1> world_rank_id=id.get_id();
          const int world_rank = world_rank_id[0];
          A[world_rank] = world_rank;
        }); // End of the kernel function
  });       // End of the queue commands
  myQueue.wait();

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
