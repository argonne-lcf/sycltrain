#include <charconv>

#ifdef __ACPP__

namespace {

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
      format += 2;  // Skip the %d
      // Adative cpp bug
      // If we put this code block in a template function
      //   template <typename T> void format_d(char *str, T value) {
      // the function will not be called, so we inline it

      // Handle zero case
      if (value == 0) {
        *str++ = '0';
      } else {
        // Count digits
        int num_digits = 0;
        T temp = value;
        while (temp > 0) {
          num_digits++;
          temp /= 10;
        }

        // Position pointer at the end of the number
        char *digit_position = str + num_digits;
        char *current = digit_position;

        // Write digits in reverse order
        temp = value;
        while (temp > 0) {
          *--current = '0' + (temp % 10);
          temp /= 10;
        }

        // Move string pointer forward
        str = digit_position;
      }

      // Process the rest of the string with remaining args
      return (str - start) + my_sprintf(str, format, args...);
    } else {
      *str++ = *format++;
    }
  }

  *str = '\0';
  return str - start;
}
}  // namespace
#endif

namespace syclx {
template <typename... Args>
void printf(const char *format, Args... args) {
#ifdef __ACPP__
  char buffr[256];
  my_sprintf(buffr, format, args...);
  sycl::detail::print(buffr);
#else
  sycl::ext::oneapi::experimental::printf(format, args...);
#endif
}
}  // namespace syclx
