#include <execution>
#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include "argparse.hpp"

int main(int argc, char **argv) {
  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |
  argparse::ArgumentParser program("6_in_order");

  program.add_argument("-g", "--global")
      .help("Global Range")
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

  //    __  _
  //   /__ |_ |\/| |\/|
  //   \_| |_ |  | |  |
  //
  sycl::queue Q{sycl::property::queue::in_order()};

  const size_t order = global_range;
  double *A = sycl::malloc_shared<double>(order * order, Q);
  double *B = sycl::malloc_shared<double>(order * order, Q);
  double *C = sycl::malloc_shared<double>(order * order, Q);

  // 0 1 2 ...
  // 0 1 2 ...
  // 0 1 2 ...
  // https://github.com/ParRes/Kernels/blob/main/Cxx11/dgemm-onemkl.cc
  for (size_t i = 0; i < order; i++) {
    for (size_t j = 0; j < order; j++) {
      A[i * order + j] = i;
      B[i * order + j] = i;
      C[i * order + j] = 0;
    }
  }

  const double alpha = 1.0;
  const double beta = 1.0;

  oneapi::mkl::blas::gemm(Q, oneapi::mkl::transpose::nontrans,  // opA
                          oneapi::mkl::transpose::nontrans,     // opB
                          order, order, order,                  // m, n, k
                          alpha,                                // alpha
                          A, order,                             // A, lda
                          B, order,                             // B, ldb
                          beta,                                 // beta
                          C, order);                            // C, ldc

  Q.wait();

  const double epsilon = 1.0e-8;
  const double reference = 0.25 * std::pow(order, 3) * std::pow(order - 1., 2);
  const double checksum =
      std::reduce(std::execution::seq, C, C + order * order, 0.);

  std::cout << "Reference checksum = " << reference << "\n"
            << "Actual checksum = " << checksum << std::endl;
  if (std::abs(checksum - reference) / reference > epsilon) {
    std::cout << "Fail" << std::endl;
    return 1;
  }
  std::cout << "Success" << std::endl;
  return 0;
}
