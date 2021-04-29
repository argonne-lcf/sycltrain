#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main() {

  //  _              _                      _
  // |_) |  _. _|_ _|_ _  ._ ._ _    ()    | \  _     o  _  _
  // |   | (_|  |_  | (_) |  | | |   (_X   |_/ (/_ \/ | (_ (/_
  //
  std::cout << "List Platforms and Devices" << std::endl;
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  for (const auto &plat : platforms) {
    // get_info is a template. So we pass the type as an `arguments`.
    std::cout << "Platform: "
              << plat.get_info<sycl::info::platform::name>() << " "
              << plat.get_info<sycl::info::platform::vendor>() << " "
              << plat.get_info<sycl::info::platform::version>() << std::endl;
    // Trivia: how can we loop over argument?

    std::vector<sycl::device> devices = plat.get_devices();
    for (const auto &dev : devices) {
      std::cout << "-- Device: "
                << dev.get_info<sycl::info::device::name>() << " "
                << (dev.is_gpu() ? "is a gpu" : " is not a gpu") << std::endl;
      // sycl::info::device::device_type exist, one can use a Map to translate to a nice sting
      // https://github.com/intel/llvm/blob/ef4e6ddce56fe21621bca7d62932713152de3f0e/sycl/source/detail/device_filter.cpp#L22
    }
  }
}
