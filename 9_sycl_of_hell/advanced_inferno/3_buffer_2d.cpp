#include <argparse.hpp>
#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

template <class T> class Matrix {
public:
  Matrix(size_t rows, size_t cols)
      : mRows(rows), mCols(cols), mData(rows * cols){};
  T &operator()(size_t i, size_t j) { return mData[i * mCols + j]; };
  T operator()(size_t i, size_t j) const { return mData[i * mCols + j]; };
  T *data() { return mData.data(); };

private:
  size_t mRows;
  size_t mCols;
  std::vector<T> mData;
};

// Inspired by Codeplay compute cpp hello-world
int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("3_buffer_2d");

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

  //  _       _   _
  // |_)    _|_ _|_ _  ._
  // |_) |_| |   | (/_ |
  //

  Matrix<int> A(global_range, global_range);

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
    // Create sycl buffer.
    // Trivia: What happend if we create the buffer in the outer scope?
    sycl::buffer<int, 2> bufferA(A.data(),
                                 sycl::range<2>(global_range, global_range));

    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA =
          bufferA.get_access<sycl::access::mode::discard_write>(cgh);
      // Nd range allow use to access information
      sycl::range<2> global(global_range, global_range);

      cgh.parallel_for<class hello_world>(
          sycl::range<2>(global), [=](sycl::nd_item<2> idx) {
            const int i = idx.get_global_id(0);
            const int j = idx.get_global_id(1);
            const int n = idx.get_global_linear_id();
            accessorA[i][j] = n;
          }); // End of the kernel function
    });       // End of the queue commands
  }           // End of scope, wait for the queued work to stop.

  for (size_t i = 0; i < global_range; i++)
    for (size_t j = 0; j < global_range; j++)
      std::cout << "A(" << i << "," << j << ") = " << A(i, j) << std::endl;
  return 0;
}
