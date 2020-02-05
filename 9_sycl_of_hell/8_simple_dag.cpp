#include "cxxopts.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

namespace sycl = cl::sycl;

template <typename TAccessorW, typename TAccessorR> class memcopy_kernel {
public:
  memcopy_kernel(TAccessorW accessorW_, TAccessorR accessorR_)
      : accessorW{accessorW_}, accessorR{accessorR_} {}
  void operator()(sycl::item<1> idx) {
    accessorW[idx.get_id()] = accessorR[idx.get_id()] + 1;
  }

private:
  TAccessorW accessorW;
  TAccessorR accessorR;
};

void f_copy(sycl::handler &cgh, int global_range,
            sycl::buffer<sycl::cl_int, 1> bufferW,
            sycl::buffer<sycl::cl_int, 1> bufferR) {
  auto accessorW = bufferW.get_access<sycl::access::mode::discard_write>(cgh);
  auto accessorR = bufferR.get_access<sycl::access::mode::read>(cgh);
  cgh.parallel_for(sycl::range<1>(global_range),
                   memcopy_kernel(accessorW, accessorR));
}

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //

  cxxopts::Options options("8_simple_dag", " How to use dag ");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"));

  auto result = options.parse(argc, argv);
  const auto global_range = result["grange"].as<int>();

  std::cout << "Running a linear dag. Output should start at 3" << std::endl;
  std::vector<int> A(global_range);
  std::iota(std::begin(A), std::end(A), 0);

  //   _      __
  //  | \ /\ /__
  //  |_//--\\_|

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";
    // A -> B -> C -> A

    // Create buffer
    sycl::buffer<sycl::cl_int, 1> bufferA(A.data(), global_range);
    sycl::buffer<sycl::cl_int, 1> bufferB(global_range);
    sycl::buffer<sycl::cl_int, 1> bufferC(global_range);

    myQueue.submit(std::bind(f_copy, std::placeholders::_1, global_range,
                             bufferB, bufferA));
    myQueue.submit(std::bind(f_copy, std::placeholders::_1, global_range,
                             bufferC, bufferB));
    myQueue.submit(std::bind(f_copy, std::placeholders::_1, global_range,
                             bufferA, bufferC));
  } // End of scope, wait for the queued work to stop.

  std::for_each(A.begin(), A.end(), [idx = 0](int v) mutable {
    std::cout << "A[ " << idx << " ] = " << v << std::endl;
    assert(v == idx + 3);
    ++idx;
  });

  return 0;
}
