#include <stdio.h>

#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("3_nd_range");

  program.add_argument("-g", "--global")
      .help("Global Range")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  program.add_argument("-l", "--local")
      .help("Local Range")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cout << program;
    std::exit(1);
  }

  const auto global_range = program.get<int>("-g");
  const auto local_range = program.get<int>("-l");

  //    __                                                           
  //   (_  |_   _. ._ _    |   _   _  _. |   |\/|  _  ._ _   _  ._   
  //   __) | | (_| | (/_   |_ (_) (_ (_| |   |  | (/_ | | | (_) | \/ 
  //                                                              /  
  sycl::queue Q;

  // Create custom shared usm allocator and use it
  sycl::usm_allocator<float, sycl::usm::alloc::shared> allocator(Q);
  std::vector<float, decltype(allocator)> A(global_range, allocator);
  std::iota(A.begin(), A.end(), 0);

  for (auto i : A) std::cout << i << ' ';
  std::cout << std::endl;

  auto A_ptr = A.data();
  Q.submit([&](sycl::handler &cgh) {
     sycl::local_accessor<float> tmp_wg(sycl::range<1>(local_range), cgh);

     cgh.parallel_for(
         sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> i) {
           int x = i.get_global_linear_id();
           int y = i.get_local_linear_id();
           tmp_wg[y] = (A_ptr[std::clamp(0, (x - 1), WORKSIZE)] + A_ptr[x] +
                     A_ptr[std::clamp(0, (x + 1), WORKSIZE)]);
           i.barrier();
           A_ptr[x] = tmp_wg[y];
         });
   }).wait();

  for (auto i : A) std::cout << i << ' ';
  std::cout << std::endl;

  return 0;
}
