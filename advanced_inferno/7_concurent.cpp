#include <chrono>
#include <locale>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

#include "argparse.hpp"

#define MAD_4(x, y) \
  x = y * x + y;    \
  y = x * y + x;    \
  x = y * x + y;    \
  y = x * y + x;
#define MAD_16(x, y) \
  MAD_4(x, y);       \
  MAD_4(x, y);       \
  MAD_4(x, y);       \
  MAD_4(x, y);
#define MAD_64(x, y) \
  MAD_16(x, y);      \
  MAD_16(x, y);      \
  MAD_16(x, y);      \
  MAD_16(x, y);

template <class T>
static T busy_wait(size_t N, T i) {
  T x = 1.3f;
  T y = i;
  for (size_t j = 0; j < N; j++) {
    MAD_64(x, y);
  }
  return y;
}

int main(int argc, char **argv) {
  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |
  argparse::ArgumentParser program("6_in_order");

  program.add_argument("-n_kernels", "-k")
      .help("num_kernel")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  program.add_argument("-n_queues", "-q")
      .help("num_queue")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  program.add_argument("-n_repetitions", "-r")
      .help("num_repetitions")
      .default_value(10)
      .action([](const std::string &value) { return std::stoi(value); });
  program.add_argument("-kernel_tripcount", "-t")
      .help("num_repetitions")
      .default_value(100000)
      .action([](const std::string &value) { return std::stoi(value); });

  program.add_argument("--in-order")
      .help("Use in-order quue, default out of order")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cout << program;
    std::exit(1);
  }

  const auto n_kernels = program.get<int>("-k");
  const auto n_queues = program.get<int>("-q");
  const auto n_repetitions = program.get<int>("-r");
  const auto kernel_tripcount = program.get<int>("-t");

  sycl::property_list pl;
  std::string q_types = "out-of-order";
  if (program["--in-order"] == true) {
    pl = sycl::property_list{sycl::property::queue::in_order{}};
    q_types = "in-order";
  }
  //    _
  //   /   _  ._   _     ._ _  ._   _
  //   \_ (_) | | (_ |_| | (/_ | | (_ \/
  //                                  /
  std::cout << "Submiting " << n_kernels << " kernels"
            << " in a round robin maner on " << n_queues << " " << q_types
            << " queues" << std::endl;

  // Shoud use KHR, default context
  const sycl::device D;
  const sycl::context C(D);

  std::vector<sycl::queue> Qs;
  for (size_t i = 0; i < n_queues; i++) Qs.push_back(sycl::queue(C, D, pl));
  // Pointer to store the value, so compiler doesn't optimize away
  auto *ptr = sycl::malloc_device<double>(n_kernels, D, C);

  // Bench
  long total_time = std::numeric_limits<long>::max();
  for (int r = 0; r < n_repetitions; r++) {
    const auto s = std::chrono::high_resolution_clock::now();
    {
      // Launch all kernels
      for (int i = 0; i < n_kernels; i++) {
        sycl::queue Q = Qs[i % n_queues];
        Q.parallel_for(sycl::range{1},
                       [=](sycl::id<1> j) {
                         ptr[i] = busy_wait(kernel_tripcount, (double)j);
                       });
      }
      // Sync
      for (auto &Q : Qs) Q.wait();
    }

    const auto e = std::chrono::high_resolution_clock::now();

    total_time = std::min(
        total_time,
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count());
  }
  std::cout.imbue(std::locale(""));
  std::cout << "Time " << total_time << "us" << std::endl;
}
