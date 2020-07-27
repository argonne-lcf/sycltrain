#include <argparse.hpp>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

class generator_kernel_hw {

public:
  generator_kernel_hw(sycl::stream cout) : m_cout(cout) {}

  void operator()(sycl::id<1> idx) {
    m_cout << "Hello, World Functor: World rank " << idx[0] << sycl::endl;
  }

private:
  sycl::stream m_cout;
};

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  argparse::ArgumentParser program("0_parallel_for_functor");

  program.add_argument("-g","--global")
   .help("Global Range")
   .default_value(1)
   .action([](const std::string& value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  const auto global_range = program.get<int>("-g");

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
