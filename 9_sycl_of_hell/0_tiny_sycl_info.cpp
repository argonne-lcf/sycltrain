#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main() {

  //  _              _                      _
  // |_) |  _. _|_ _|_ _  ._ ._ _    ()    | \  _     o  _  _
  // |   | (_|  |_  | (_) |  | | |   (_X   |_/ (/_ \/ | (_ (/_
  //
  std::cout << "List Platforms and Devices" << std::endl;
  std::vector<sycl::platform> platforms = cl::sycl::platform::get_platforms();
  for (const auto &plat : platforms) {
    // get_info is a template. So we pass the type as an `arguments`.
    std::cout << "Platform: ";

    std::cout << plat.get_info<sycl::info::platform::name>() << " ";
    std::cout << plat.get_info<sycl::info::platform::vendor>() << " ";
    std::cout << plat.get_info<sycl::info::platform::version>() << std::endl;
    // Trivia: how can we loop over argument?

    std::vector<cl::sycl::device> devices = plat.get_devices();
    for (const auto &dev : devices) {
      std::cout << "-- Device: ";
      std::cout << dev.get_info<sycl::info::device::name>() << " ";
      std::cout << (dev.is_gpu() ? "is a gpu" : " is not a gpu") << std::endl;
      // sycl::info::device::device_type exist, but do not overload the <<
      // operator
    }
  }
}
