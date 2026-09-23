// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Feature probes: the standard library of C++26. Each probe is compiled,
// linked and run on its own by run.py, and passes when it exits with status 0.
// The probes follow the project's rules for C++: an empty parameter list is
// written (void), and no conversion is left implicit.

//=== probe: simd_vec
//--- title: std::simd::vec (<simd>, C++26 names)
//--- paper: P1928R15, P3287R3
//--- std: c++26
//--- macro: __cpp_lib_simd
#include <cstddef>
#include <simd>
int main(void) {
  std::simd::vec<float> a(1.5f);
  std::simd::vec<float> b([](auto i) -> float { return static_cast<float>(i); });
  auto c = a + b;
  float s = std::simd::reduce(c);
  float expect = 0.0f;
  for (std::size_t i = 0uz; i < static_cast<std::size_t>(c.size()); ++i) expect += 1.5f + static_cast<float>(i);
  return s == expect ? 0 : 1;
}

//=== probe: simd_draft_names
//--- title: std::simd<float> (<simd>, names of P1928 before P3287)
//--- paper: P1928R15
//--- std: c++26
#include <simd>
int main(void) {
  std::simd<float> a(1.5f);
  std::simd<float> b([](auto i) -> float { return static_cast<float>(i); });
  auto c = a + b;
  float s = std::reduce(c);
  return s > 0.0f ? 0 : 1;
}

//=== probe: experimental_simd
//--- title: std::experimental::simd (Parallelism TS 2)
//--- paper: N4808
//--- macro: __cpp_lib_experimental_parallel_simd
#include <cstddef>
#include <experimental/simd>
namespace stdx = std::experimental;
int main(void) {
  stdx::native_simd<float> a = 1.5f;
  stdx::native_simd<float> b([](auto i) -> float { return static_cast<float>(i); });
  auto c = a + b;
  float s = stdx::reduce(c);
  float expect = 0.0f;
  for (std::size_t i = 0uz; i < c.size(); ++i) expect += 1.5f + static_cast<float>(i);
  return s == expect ? 0 : 1;
}

//=== probe: submdspan
//--- title: std::submdspan, full_extent, strided_slice
//--- paper: P2630R4
//--- std: c++26
//--- macro: __cpp_lib_submdspan
#include <mdspan>
#include <utility>
int main(void) {
  float buf[12]{};
  std::mdspan m(buf, 3uz, 4uz);
  auto row = std::submdspan(m, 1uz, std::full_extent);
  auto block = std::submdspan(m, std::pair{1uz, 3uz}, std::strided_slice{0uz, 4uz, 2uz});
  row[2uz] = 5.0f;
  return buf[1 * 4 + 2] == 5.0f && row.extent(0uz) == 4uz && block.extent(0uz) == 2uz && block.extent(1uz) == 2uz
             ? 0
             : 1;
}

//=== probe: mdspan_padded
//--- title: std::layout_right_padded / layout_left_padded
//--- paper: P2642R6
//--- std: c++26
#include <mdspan>
int main(void) {
  float buf[3 * 8]{};
  using E = std::dextents<int, 2>;
  std::layout_right_padded<8>::mapping<E> map(E{3, 5});
  std::mdspan<float, E, std::layout_right_padded<8>> m(buf, map);
  m[2, 4] = 1.0f;
  return buf[2 * 8 + 4] == 1.0f && map.stride(0) == 8 ? 0 : 1;
}

//=== probe: aligned_accessor
//--- title: std::aligned_accessor for mdspan
//--- paper: P2897R7
//--- std: c++26
//--- macro: __cpp_lib_aligned_accessor
#include <cstddef>
#include <mdspan>
int main(void) {
  alignas(64) float buf[16]{};
  using E = std::dextents<std::size_t, 1>;
  std::mdspan<float, E, std::layout_right, std::aligned_accessor<float, 64>> m(buf, 16uz);
  m[3uz] = 2.0f;
  return buf[3] == 2.0f ? 0 : 1;
}

//=== probe: dims
//--- title: std::dims
//--- paper: P2389R2
//--- std: c++26
#include <cstddef>
#include <mdspan>
#include <type_traits>
int main(void) {
  static_assert(std::is_same_v<std::dims<2>, std::dextents<std::size_t, 2>>);
  return 0;
}

//=== probe: saturation_arithmetic
//--- title: add_sat, sub_sat, saturate_cast
//--- paper: P0543R3
//--- std: c++26
//--- macro: __cpp_lib_saturation_arithmetic
#include <cstdint>
#include <numeric>
int main(void) {
  static_assert(std::add_sat<std::uint8_t>(200, 100) == 255);
  static_assert(std::sub_sat<std::int16_t>(-32000, 1000) == -32768);
  static_assert(std::saturate_cast<std::uint8_t>(-5) == 0);
  static_assert(std::saturate_cast<std::uint8_t>(300) == 255);
  return 0;
}

//=== probe: inplace_vector
//--- title: std::inplace_vector
//--- paper: P0843R14
//--- std: c++26
//--- macro: __cpp_lib_inplace_vector
#include <inplace_vector>
int main(void) {
  std::inplace_vector<int, 4> v;
  v.push_back(1);
  v.push_back(2);
  int* p = v.try_push_back(3);
  return v.size() == 3uz && p != nullptr && *p == 3 && v.capacity() == 4uz ? 0 : 1;
}

//=== probe: linalg
//--- title: std::linalg (BLAS-like on mdspan)
//--- paper: P1673R13
//--- std: c++26
//--- macro: __cpp_lib_linalg
#include <linalg>
#include <mdspan>
int main(void) {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float x[2] = {1.0f, 1.0f};
  float y[2] = {};
  std::mdspan A(a, 2uz, 2uz);
  std::mdspan X(x, 2uz);
  std::mdspan Y(y, 2uz);
  std::linalg::matrix_vector_product(A, X, Y);
  return y[0] == 3.0f && y[1] == 7.0f ? 0 : 1;
}

//=== probe: views_concat
//--- title: views::concat
//--- paper: P2542R8
//--- std: c++26
//--- macro: __cpp_lib_ranges_concat
#include <ranges>
#include <vector>
int main(void) {
  std::vector<int> a{1, 2};
  std::vector<int> b{3};
  int s = 0;
  for (int x : std::views::concat(a, b)) s += x;
  return s == 6 ? 0 : 1;
}

//=== probe: views_indices
//--- title: views::indices
//--- paper: P3060R3
//--- std: c++26
//--- macro: __cpp_lib_ranges_indices
#include <ranges>
int main(void) {
  int s = 0;
  for (int i : std::views::indices(4)) s += i;
  return s == 6 ? 0 : 1;
}

//=== probe: function_ref
//--- title: std::function_ref
//--- paper: P0792R14
//--- std: c++26
//--- macro: __cpp_lib_function_ref
#include <functional>
static int apply(std::function_ref<int(int)> f) { return f(2); }
int main(void) {
  int k = 3;
  return apply([&](int x) -> int { return x * k; }) == 6 ? 0 : 1;
}

//=== probe: copyable_function
//--- title: std::copyable_function
//--- paper: P2548R6
//--- std: c++26
//--- macro: __cpp_lib_copyable_function
#include <functional>
int main(void) {
  std::copyable_function<int(int) const> f = [](int x) -> int { return x + 1; };
  auto g = f;
  return g(1) == 2 ? 0 : 1;
}

//=== probe: optional_ref
//--- title: std::optional<T&>
//--- paper: P2988R12
//--- std: c++26
#include <optional>
int main(void) {
  int x = 1;
  std::optional<int&> r = x;
  *r = 5;
  return x == 5 ? 0 : 1;
}

//=== probe: optional_range
//--- title: std::optional as a range
//--- paper: P3168R2
//--- std: c++26
//--- macro: __cpp_lib_optional_range_support
#include <optional>
int main(void) {
  std::optional<int> o = 3;
  int s = 0;
  for (int v : o) s += v;
  return s == 3 ? 0 : 1;
}

//=== probe: runtime_format
//--- title: std::runtime_format
//--- paper: P2918R2
//--- std: c++26
#include <format>
#include <string>
int main(void) {
  std::string fmt = "{}-{}";
  return std::format(std::runtime_format(fmt), 1, 2) == "1-2" ? 0 : 1;
}

//=== probe: println_blank
//--- title: std::println() with no arguments
//--- paper: P3142R0
//--- std: c++26
#include <print>
int main(void) {
  std::println();
  return 0;
}

//=== probe: cxx_stdckdint
//--- title: <stdckdint.h> ckd_add / ckd_mul in C++
//--- paper: P3370R1
//--- std: c++26
//--- macro: __cpp_lib_stdckdint
#include <cstddef>
#include <stdckdint.h>
int main(void) {
  std::size_t bytes = 0uz;
  bool overflow = ckd_mul(&bytes, std::size_t{1} << 40, std::size_t{1} << 40);
  return overflow ? 0 : 1;
}

//=== probe: text_encoding
//--- title: std::text_encoding
//--- paper: P1885R12
//--- std: c++26
//--- macro: __cpp_lib_text_encoding
#include <text_encoding>
int main(void) {
  auto literal = std::text_encoding::literal();
  return literal.mib() != std::text_encoding::id::unknown ? 0 : 1;
}

//=== probe: debugging
//--- title: std::is_debugger_present / breakpoint
//--- paper: P2546R5
//--- std: c++26
//--- macro: __cpp_lib_debugging
#include <debugging>
int main(void) {
  bool present = std::is_debugger_present();
  static_cast<void>(present);
  return 0;
}

//=== probe: execution_senders
//--- title: std::execution senders and receivers
//--- paper: P2300R10
//--- std: c++26
//--- macro: __cpp_lib_senders
#include <execution>
#include <utility>
int main(void) {
  namespace ex = std::execution;
  auto work = ex::just(20) | ex::then([](int v) -> int { return v + 22; });
  auto [r] = std::this_thread::sync_wait(std::move(work)).value();
  return r == 42 ? 0 : 1;
}

//=== probe: hive
//--- title: std::hive
//--- paper: P0447R28
//--- std: c++26
//--- macro: __cpp_lib_hive
#include <hive>
int main(void) {
  std::hive<int> h;
  auto it = h.insert(1);
  h.insert(2);
  h.erase(it);
  return h.size() == 1uz ? 0 : 1;
}

//=== probe: philox_engine
//--- title: std::philox_engine (counter-based random numbers)
//--- paper: P2075R6
//--- std: c++26
//--- macro: __cpp_lib_philox_engine
#include <random>
int main(void) {
  std::philox4x32 e(42u);
  std::philox4x32 f(42u);
  return e() == f() ? 0 : 1;
}

//=== probe: is_within_lifetime
//--- title: std::is_within_lifetime
//--- paper: P2641R4
//--- std: c++26
//--- macro: __cpp_lib_is_within_lifetime
#include <type_traits>
constexpr bool alive(void) {
  int x = 0;
  return std::is_within_lifetime(&x);
}
static_assert(alive());
int main(void) { return 0; }

//=== probe: span_at
//--- title: std::span::at
//--- paper: P2821R5
//--- std: c++26
//--- macro: __cpp_lib_span
#include <span>
#include <stdexcept>
int main(void) {
  int a[3]{1, 2, 3};
  std::span s(a);
  try {
    static_cast<void>(s.at(3uz));
    return 1;
  } catch (const std::out_of_range&) {
  }
  return s.at(1uz) == 2 ? 0 : 1;
}

//=== probe: atomic_fetch_max
//--- title: atomic fetch_max / fetch_min
//--- paper: P0493R5
//--- std: c++26
//--- macro: __cpp_lib_atomic_min_max
#include <atomic>
int main(void) {
  std::atomic<int> a{3};
  a.fetch_max(7);
  a.fetch_min(5);
  return a.load() == 5 ? 0 : 1;
}
