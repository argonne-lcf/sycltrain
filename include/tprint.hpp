#include <charconv>

namespace {
template <typename T, typename... Args> void format_d(char *str, T value) {
  // Handle zero case
  if (value == 0) {
    *str++ = '0';
    return;
  }

  // Convert integer to string using std::to_chars.
  // This may overflow, and all the good stuff.
  auto [ptr, ec] = std::to_chars(str, str + 32, value);
  if (ec == std::errc()) {
    str = ptr;
  }
}

// Base case for recursion
int my_sprintf(char *str, const char *format) {
  char *start = str;
  while (*format) {
    *str++ = *format++;
  }
  *str = '\0';
  return str - start;
}

// Template version for handling integer replacement
template <typename T, typename... Args>
int my_sprintf(char *str, const char *format, T value, Args... args) {
  char *start = str;

  while (*format) {
    // Handle the %d format for integers
    if (*format == '%' && *(format + 1) == 'd') {
      format += 2; // Skip the %d
      format_d(str, value);
      // Process the rest of the string with remaining args
      return (str - start) + my_sprintf(str, format, args...);
    } else {
      *str++ = *format++;
    }
  }

  *str = '\0';
  return str - start;
}
}

namespace syclx {
template <typename... Args> void printf(const char *format, Args... args) {
  #ifdef __ACPP__
  char buffr[256];
  my_sprintf(buffr, format, args...);
  sycl::detail::print(buf);
  #else
  sycl::ext::oneapi::experimental::printf(format, args...);
  #endif
}
}
