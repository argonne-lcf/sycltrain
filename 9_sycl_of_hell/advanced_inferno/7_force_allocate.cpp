#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

// How to transform this function into a variadic one
template <sycl::access::target Q, typename T, int I>
void force_allocate(sycl::buffer<T, I> buffer, sycl::queue myQueue) {

  // Create a device accesors and use it. This will force the allocation
  myQueue.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, I, sycl::access::mode::write, Q> accessorA(
        buffer, cgh, buffer.get_size());
    cgh.single_task<class allocate>([=]() { accessorA[0]; });
  });
  myQueue.wait();
}

int main(int argc, char **argv) {

  const auto global_range = (size_t)atoi(argv[1]);
  std::vector<int> A(global_range);

  sycl::default_selector selector;
  {
    sycl::queue myQueue(selector);

    // Create buffer.
    sycl::buffer<sycl::cl_int, 1> bufferA(A.data(), global_range);

    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Force the allocation of the buffer.

    force_allocate<sycl::access::target::global_buffer>(bufferA, myQueue);
  } // End of scope, wait for the queued work to stop.
  return 0;
}
