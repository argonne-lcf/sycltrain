#include <sycl/sycl.hpp>
#include <experimental/mdspan>

#include "argparse.hpp"

int main(int argc, char **argv) {
  //
  //   |\/|  _|    _ ._   _. ._
  //   |  | (_|   _> |_) (_| | |
  //                 |
  // Port of https://en.cppreference.com/w/cpp/container/mdspan
  sycl::queue Q;
  sycl::usm_allocator<int, sycl::usm::alloc::shared> allocator(Q);
  std::vector<int, decltype(allocator)> v(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, allocator);
  // View data as contiguous memory representing 2 rows of 6 ints each
  auto ms2 = std::mdspan(v.data(), 2, 6);
  // View the same data as a 3D array 2 x 3 x 2
  auto ms3 = std::mdspan(v.data(), 2, 3, 2);

  // Write data using 2D view on the gpu
  Q.parallel_for(ms2.extent(0), [=](auto i) {
     for (std::size_t j = 0; j != ms2.extent(1); j++)
       ms2(static_cast<int>(i), j) = i * 1000 + j;
   }).wait();

  // Read back using 3D view
  for (std::size_t i = 0; i != ms3.extent(0); i++) {
    std::cout << "slice @ i = " << i << std::endl;
    for (std::size_t j = 0; j != ms3.extent(1); j++) {
      for (std::size_t k = 0; k != ms3.extent(2); k++)
        std::cout << ms3(i, j, k) << " ";
      std::cout << std::endl;
    }
  }
}
