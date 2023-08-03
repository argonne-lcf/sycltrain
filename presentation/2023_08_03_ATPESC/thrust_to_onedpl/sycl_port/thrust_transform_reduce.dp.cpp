#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <functional>
#include <cmath>

// to compile: clang++ -std=c++17 -sycl-std=2020 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DONEDPL_USE_TBB_BACKEND=OFF thrust_transform_reduce.dp.cpp -o thrust_transform_reduce.dp.out

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    
        T operator()(const T& x) const {
            return x * x;
        }
};

int main(void)
{
    // initialize host array
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    dpct::device_vector<float> d_x(x, x + 4);

    // setup arguments
    square<float>        unary_op;
    std::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt(std::transform_reduce(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        d_x.begin(), d_x.end(), init, binary_op, unary_op));

    std::cout << norm << std::endl;

    return 0;
}
