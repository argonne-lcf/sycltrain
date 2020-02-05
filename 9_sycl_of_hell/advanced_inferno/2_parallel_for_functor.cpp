#include "cxxopts.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

class generator_kernel_hw {

public:
  generator_kernel_hw(sycl::stream cout) : m_cout(cout) {}

  void operator()(sycl::id<1> idx) {
    m_cout << "Hello, World Functor: World rank " << idx << sycl::endl;
  }

private:
  sycl::stream m_cout;
};
int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  cxxopts::Options options("2_parallel_for",
                           " How to use functor and not lambda ");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"));

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const auto global_range = result["grange"].as<int>();

  //  _                             _
  // |_) _. ._ ._ _. | | |  _  |   |_ _  ._
  // |  (_| |  | (_| | | | (/_ |   | (_) |

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  {

    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    myQueue.submit([&](sycl::handler &cgh) {
      sycl::stream cout(1024, 256, cgh);
      auto hw_kernel = generator_kernel_hw(cout);
      cgh.parallel_for(sycl::range<1>(global_range), hw_kernel);
    }); // End of the queue commands
  }     // End of scope, wait for the queued work to stop.
  return 0;
}
