#include <stdio.h>

#include <iostream>
#include <sycl/sycl.hpp>

#define WORKSIZE 32
#define WORKITEM 16

int main(int argc, char **argv) {
  sycl::queue Q;

  // Create custom shared usm allocator and use it
  sycl::usm_allocator<float, sycl::usm::alloc::shared> allocator(Q);
  std::vector<float, decltype(allocator)> A(WORKSIZE, allocator);
  std::iota(A.begin(), A.end(), 0);

  for (auto i : A) std::cout << i << ' ';
  std::cout << std::endl;

  auto A_ptr = A.data();
  Q.submit([&](sycl::handler &cgh) {
     sycl::local_accessor<float, 1> acc(sycl::range<1>(WORKITEM), cgh);

     cgh.parallel_for(
         sycl::nd_range<1>(WORKSIZE, WORKITEM), [=](sycl::nd_item<1> i) {
           int x = i.get_global_linear_id();
           int y = i.get_local_linear_id();
           acc[y] = (A_ptr[std::clamp(0, (x - 1), WORKSIZE)] + A_ptr[x] +
                     A_ptr[std::clamp(0, (x + 1), WORKSIZE)]);
           i.barrier();
           A_ptr[x] = acc[y];
         });
   }).wait();

  for (auto i : A) std::cout << i << ' ';
  std::cout << std::endl;

  return 0;
}
