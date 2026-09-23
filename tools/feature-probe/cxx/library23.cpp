// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Feature probes: the standard library of C++23, and the parts of C++17/20
// whose support differs between the three standard libraries (MSVC STL,
// libstdc++, Apple's libc++). Each probe is compiled, linked and run on its own
// by run.py, and passes when it exits with status 0. The probes follow the
// project's rules for C++: an empty parameter list is written (void), and no
// conversion is left implicit.

//=== probe: mdspan
//--- title: std::mdspan, extents, layout_right / layout_stride
//--- paper: P0009R18
//--- macro: __cpp_lib_mdspan
#include <array>
#include <mdspan>
#include <vector>
int main(void) {
  std::vector<float> buf(6uz);
  std::mdspan m(buf.data(), 2uz, 3uz);
  m[1uz, 2uz] = 5.0f;
  std::mdspan<float, std::extents<int, 2, 3>> fixed(buf.data());
  using E = std::dextents<int, 2>;
  std::layout_stride::mapping<E> column_major(E{2, 3}, std::array<int, 2>{1, 2});
  std::mdspan<float, E, std::layout_stride> cm(buf.data(), column_major);
  return buf[5] == 5.0f && fixed[1, 2] == 5.0f && m.extent(1uz) == 3uz && &cm[1, 2] == &buf[1 + 2 * 2] ? 0 : 1;
}

//=== probe: expected
//--- title: std::expected with monadic operations
//--- paper: P0323R12, P2505R5
//--- macro: __cpp_lib_expected
#include <expected>
#include <string>
static std::expected<int, std::string> parse(int x) {
  if (x < 0) return std::unexpected(std::string("negative"));
  return x;
}
int main(void) {
  auto r = parse(2)
               .and_then([](int v) -> std::expected<int, std::string> { return v * 2; })
               .transform([](int v) -> int { return v + 1; });
  auto e = parse(-1).transform_error([](const std::string& s) -> std::size_t { return s.size(); });
  std::expected<void, int> ok{};
  return r.has_value() && *r == 5 && !e.has_value() && e.error() == 8uz && ok.has_value() ? 0 : 1;
}

//=== probe: print
//--- title: std::print / std::println
//--- paper: P2093R14
//--- macro: __cpp_lib_print
#include <cstdio>
#include <print>
int main(void) {
  std::println("probe {} {:.3f}", 1, 2.5);
  std::print(stderr, "to stderr\n");
  std::println(stdout, "{:>5}", "ab");
  return 0;
}

//=== probe: format_ranges
//--- title: formatting ranges with std::format
//--- paper: P2286R8
//--- macro: __cpp_lib_format_ranges
#include <format>
#include <map>
#include <string>
#include <vector>
int main(void) {
  std::vector<int> v{1, 2};
  std::map<int, int> m{{1, 2}};
  return std::format("{}", v) == "[1, 2]" && std::format("{}", m) == "{1: 2}" ? 0 : 1;
}

//=== probe: ranges_to
//--- title: std::ranges::to and container append_range
//--- paper: P1206R7
//--- macro: __cpp_lib_ranges_to_container, __cpp_lib_containers_ranges
#include <ranges>
#include <set>
#include <vector>
int main(void) {
  auto v = std::views::iota(0, 5) | std::ranges::to<std::vector>();
  std::vector<int> w;
  w.append_range(v);
  auto s = v | std::ranges::to<std::set<int>>();
  return v.size() == 5uz && w.size() == 5uz && s.size() == 5uz ? 0 : 1;
}

//=== probe: views_zip
//--- title: views::zip, zip_transform, adjacent, pairwise_transform
//--- paper: P2321R2
//--- macro: __cpp_lib_ranges_zip
#include <functional>
#include <ranges>
#include <vector>
int main(void) {
  std::vector<int> a{1, 2, 3};
  std::vector<float> b{0.5f, 1.5f, 2.5f};
  float s = 0.0f;
  for (auto [x, y] : std::views::zip(a, b)) s += static_cast<float>(x) * y;
  auto sums = std::views::zip_transform(std::plus<>{}, a, a);
  auto diffs = a | std::views::pairwise_transform([](int p, int q) -> int { return q - p; });
  auto adjacent = a | std::views::adjacent<2>;
  return s == 11.0f && sums[2] == 6 && diffs[0] == 1 && std::ranges::distance(adjacent) == 2 ? 0 : 1;
}

//=== probe: views_enumerate
//--- title: views::enumerate
//--- paper: P2164R9
//--- macro: __cpp_lib_ranges_enumerate
#include <cstddef>
#include <ranges>
#include <vector>
int main(void) {
  std::vector<int> v{5, 6, 7};
  std::ptrdiff_t s = 0;
  for (auto [i, x] : std::views::enumerate(v)) s += i * static_cast<std::ptrdiff_t>(x);
  return s == 20 ? 0 : 1;
}

//=== probe: views_cartesian_product
//--- title: views::cartesian_product
//--- paper: P2374R4
//--- macro: __cpp_lib_ranges_cartesian_product
#include <ranges>
int main(void) {
  int count = 0;
  int last = -1;
  for (auto [i, j] : std::views::cartesian_product(std::views::iota(0, 3), std::views::iota(0, 4))) {
    ++count;
    last = i * 4 + j;
  }
  return count == 12 && last == 11 ? 0 : 1;
}

//=== probe: views_chunk_slide_stride
//--- title: views::chunk, chunk_by, slide, stride
//--- paper: P2442R1, P2443R1, P1899R3
//--- macro: __cpp_lib_ranges_chunk, __cpp_lib_ranges_slide, __cpp_lib_ranges_stride, __cpp_lib_ranges_chunk_by
#include <functional>
#include <ranges>
#include <vector>
int main(void) {
  auto r = std::views::iota(0, 10);
  auto chunks = r | std::views::chunk(4);
  auto windows = r | std::views::slide(3);
  auto strided = r | std::views::stride(3);
  std::vector<int> runs{1, 1, 2, 2, 2, 3};
  auto groups = runs | std::views::chunk_by(std::ranges::equal_to{});
  return std::ranges::distance(chunks) == 3 && std::ranges::distance(windows) == 8 &&
                 std::ranges::distance(strided) == 4 && std::ranges::distance(groups) == 3
             ? 0
             : 1;
}

//=== probe: views_join_with_repeat
//--- title: views::join_with, repeat, as_rvalue, as_const
//--- paper: P2441R2, P2474R2, P2446R2, P2278R4
//--- macro: __cpp_lib_ranges_join_with, __cpp_lib_ranges_repeat, __cpp_lib_ranges_as_rvalue, __cpp_lib_ranges_as_const
#include <ranges>
#include <string>
#include <utility>
#include <vector>
int main(void) {
  std::vector<std::string> words{"a", "b", "c"};
  std::string joined;
  for (char c : words | std::views::join_with(',')) joined += c;
  auto rep = std::views::repeat(7, 3);
  std::vector<std::string> moved;
  for (auto&& s : words | std::views::as_rvalue) moved.push_back(std::forward<decltype(s)>(s));
  auto view = moved | std::views::as_const;
  return joined == "a,b,c" && std::ranges::distance(rep) == 3 && view[1] == "b" ? 0 : 1;
}

//=== probe: ranges_fold
//--- title: ranges::fold_left, fold_left_first, fold_right
//--- paper: P2322R6
//--- macro: __cpp_lib_ranges_fold
#include <algorithm>
#include <functional>
#include <vector>
int main(void) {
  std::vector<int> v{1, 2, 3, 4};
  int s = std::ranges::fold_left(v, 0, std::plus{});
  auto m = std::ranges::fold_left_first(v, [](int a, int b) -> int { return a > b ? a : b; });
  int r = std::ranges::fold_right(v, 0, std::minus{});
  return s == 10 && m.has_value() && *m == 4 && r == -2 ? 0 : 1;
}

//=== probe: ranges_contains
//--- title: ranges::contains, contains_subrange
//--- paper: P2302R4
//--- macro: __cpp_lib_ranges_contains
#include <algorithm>
#include <vector>
int main(void) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> mid{2, 3};
  return std::ranges::contains(v, 3) && std::ranges::contains_subrange(v, mid) ? 0 : 1;
}

//=== probe: ranges_starts_ends_with
//--- title: ranges::starts_with, ends_with
//--- paper: P1659R3
//--- macro: __cpp_lib_ranges_starts_ends_with
#include <algorithm>
#include <vector>
int main(void) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> head{1, 2};
  std::vector<int> tail{4, 5};
  return std::ranges::starts_with(v, head) && std::ranges::ends_with(v, tail) ? 0 : 1;
}

//=== probe: ranges_find_last
//--- title: ranges::find_last
//--- paper: P1223R5
//--- macro: __cpp_lib_ranges_find_last
#include <algorithm>
#include <vector>
int main(void) {
  std::vector<int> v{2, 1, 2, 3};
  auto last = std::ranges::find_last(v, 2);
  return !last.empty() && last.begin() == v.begin() + 2 ? 0 : 1;
}

//=== probe: ranges_iota
//--- title: ranges::iota
//--- paper: P2440R1
//--- macro: __cpp_lib_ranges_iota
#include <numeric>
#include <vector>
int main(void) {
  std::vector<int> v(5uz);
  std::ranges::iota(v, 1);
  return v[4] == 5 ? 0 : 1;
}

//=== probe: optional_monadic
//--- title: std::optional and_then / transform / or_else
//--- paper: P0798R8
//--- macro: __cpp_lib_optional
#include <optional>
int main(void) {
  std::optional<int> o = 2;
  auto r = o.and_then([](int v) -> std::optional<int> { return v * 3; })
               .transform([](int v) -> int { return v + 1; })
               .or_else([](void) -> std::optional<int> { return 0; });
  return r == 7 ? 0 : 1;
}

//=== probe: move_only_function
//--- title: std::move_only_function
//--- paper: P0288R9
//--- macro: __cpp_lib_move_only_function
#include <functional>
#include <memory>
#include <utility>
int main(void) {
  auto p = std::make_unique<int>(3);
  std::move_only_function<int(void)> f = [q = std::move(p)](void) -> int { return *q; };
  return f() == 3 ? 0 : 1;
}

//=== probe: out_ptr
//--- title: std::out_ptr / inout_ptr for C APIs
//--- paper: P1132R8
//--- macro: __cpp_lib_out_ptr
#include <memory>
static void make(int** out) { *out = new int(7); }
int main(void) {
  std::unique_ptr<int> p;
  make(std::out_ptr(p));
  return p != nullptr && *p == 7 ? 0 : 1;
}

//=== probe: stacktrace
//--- title: std::stacktrace
//--- paper: P0881R7
//--- macro: __cpp_lib_stacktrace
//--- libs:  | -lstdc++exp | -lstdc++_libbacktrace
#include <stacktrace>
int main(void) {
  auto trace = std::stacktrace::current();
  return trace.size() > 0uz ? 0 : 1;
}

//=== probe: generator
//--- title: std::generator
//--- paper: P2502R2
//--- macro: __cpp_lib_generator
#include <generator>
static std::generator<int> count_to(int n) {
  for (int i = 0; i < n; ++i) co_yield i;
}
int main(void) {
  int s = 0;
  for (int v : count_to(5)) s += v;
  return s == 10 ? 0 : 1;
}

//=== probe: flat_map
//--- title: std::flat_map / std::flat_set
//--- paper: P0429R9, P1222R4
//--- macro: __cpp_lib_flat_map, __cpp_lib_flat_set
#include <flat_map>
#include <flat_set>
int main(void) {
  std::flat_map<int, int> m;
  m[3] = 1;
  m[1] = 2;
  std::flat_set<int> s{3, 1, 2};
  return m.begin()->first == 1 && *s.begin() == 1 ? 0 : 1;
}

//=== probe: spanstream
//--- title: std::spanstream
//--- paper: P0448R4
//--- macro: __cpp_lib_spanstream
#include <span>
#include <spanstream>
int main(void) {
  char buf[16]{};
  std::ospanstream os{std::span<char>(buf)};
  os << 42;
  return buf[0] == '4' && buf[1] == '2' ? 0 : 1;
}

//=== probe: utility23
//--- title: unreachable, to_underlying, byteswap, forward_like, invoke_r
//--- paper: P0627R6, P1682R3, P1272R4, P2445R1, P2136R3
//--- macro: __cpp_lib_unreachable, __cpp_lib_to_underlying, __cpp_lib_byteswap, __cpp_lib_forward_like, __cpp_lib_invoke_r
#include <bit>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
enum class Kind : std::uint8_t { a = 3 };
int main(int argc, char**) {
  if (argc < 0) std::unreachable();
  static_assert(std::to_underlying(Kind::a) == 3);
  static_assert(std::byteswap(std::uint32_t{0x11223344u}) == 0x44332211u);
  int x = 1;
  static_assert(std::is_same_v<decltype(std::forward_like<const int&>(x)), const int&>);
  long r = std::invoke_r<long>([](void) -> int { return 5; });
  return r == 5L ? 0 : 1;
}

//=== probe: constexpr_cmath
//--- title: constexpr <cmath> basics (fabs, floor, fmax, copysign, isnan)
//--- paper: P0533R9
//--- macro: __cpp_lib_constexpr_cmath
#include <cmath>
static_assert(std::fabs(-1.5) == 1.5);
static_assert(std::floor(1.5) == 1.0);
static_assert(std::fmax(1.0, 2.0) == 2.0);
static_assert(std::copysign(1.0, -0.0) == -1.0);
static_assert(!std::isnan(1.0));
int main(void) { return 0; }

//=== probe: string23
//--- title: string::contains, resize_and_overwrite
//--- paper: P1679R3, P1072R10
//--- macro: __cpp_lib_string_contains, __cpp_lib_string_resize_and_overwrite
#include <cstddef>
#include <string>
int main(void) {
  std::string s = "hello";
  s.resize_and_overwrite(10uz, [](char* p, std::size_t) -> std::size_t {
    p[5] = '!';
    return 6uz;
  });
  return s.contains("lo!") && s.size() == 6uz ? 0 : 1;
}

//=== probe: start_lifetime_as
//--- title: std::start_lifetime_as
//--- paper: P2590R2
//--- macro: __cpp_lib_start_lifetime_as
#include <cstring>
#include <memory>
struct Pod {
  int x;
  float y;
};
int main(void) {
  alignas(Pod) unsigned char buf[sizeof(Pod)];
  Pod src{1, 2.0f};
  std::memcpy(buf, &src, sizeof src);
  Pod* p = std::start_lifetime_as<Pod>(buf);
  return p->x == 1 ? 0 : 1;
}

//=== probe: bind_back
//--- title: std::bind_back
//--- paper: P2387R3
//--- macro: __cpp_lib_bind_back
#include <functional>
int main(void) {
  auto f = std::bind_back([](int a, int b) -> int { return a - b; }, 1);
  return f(3) == 2 ? 0 : 1;
}

//=== probe: constexpr_unique_ptr
//--- title: constexpr std::unique_ptr
//--- paper: P2273R3
//--- macro: __cpp_lib_constexpr_memory
#include <memory>
constexpr int four(void) {
  auto p = std::make_unique<int>(4);
  return *p;
}
static_assert(four() == 4);
int main(void) { return 0; }

//=== probe: threads20
//--- title: jthread, stop_token, barrier, latch, counting_semaphore, atomic wait
//--- paper: P0660R10, P1135R6
//--- std: c++20
//--- macro: __cpp_lib_jthread, __cpp_lib_barrier, __cpp_lib_latch, __cpp_lib_semaphore, __cpp_lib_atomic_wait
#include <atomic>
#include <barrier>
#include <latch>
#include <semaphore>
#include <stop_token>
#include <thread>
#include <vector>
int main(void) {
  constexpr int n = 4;
  std::atomic<int> phases{0};
  std::barrier sync(n, [&](void) noexcept -> void { phases.fetch_add(1); });
  std::latch done(n);
  std::counting_semaphore<n> slots(0);
  std::vector<std::jthread> threads;
  for (int i = 0; i < n; ++i)
    threads.emplace_back([&](std::stop_token token) -> void {
      sync.arrive_and_wait();
      sync.arrive_and_wait();
      slots.release();
      done.count_down();
      static_cast<void>(token.stop_requested());
    });
  done.wait();
  for (int i = 0; i < n; ++i) slots.acquire();
  std::atomic<int> flag{0};
  std::jthread waiter([&](void) -> void { flag.wait(0); });
  flag.store(1);
  flag.notify_one();
  waiter.join();
  return phases.load() == 2 ? 0 : 1;
}

//=== probe: format_float
//--- title: std::format of floating point (needs to_chars in the library)
//--- paper: P0645R10
//--- std: c++20
//--- macro: __cpp_lib_format
#include <format>
#include <string>
int main(void) {
  std::string s = std::format("{:.3f} {:g} {}", 3.14159, 1e-5, 0.1);
  return s == "3.142 1e-05 0.1" ? 0 : 1;
}

//=== probe: charconv_float
//--- title: std::from_chars / to_chars for double and float
//--- paper: P0067R5
//--- std: c++20
//--- macro: __cpp_lib_to_chars
#include <charconv>
#include <string_view>
#include <system_error>
int main(void) {
  constexpr std::string_view in = "1.25e-3";
  double d = 0.0;
  auto r = std::from_chars(in.data(), in.data() + in.size(), d);
  constexpr std::string_view in2 = "2.5";
  float f = 0.0f;
  auto r2 = std::from_chars(in2.data(), in2.data() + in2.size(), f);
  char out[32];
  auto w = std::to_chars(out, out + sizeof out, 0.5);
  return r.ec == std::errc{} && d == 1.25e-3 && r2.ec == std::errc{} && f == 2.5f && w.ec == std::errc{} &&
                 std::string_view(out, w.ptr) == "0.5"
             ? 0
             : 1;
}

//=== probe: filesystem_utf8
//--- title: std::filesystem with UTF-8 (u8) file names
//--- paper: P0218R1, P0482R6
//--- std: c++20
//--- macro: __cpp_lib_filesystem, __cpp_lib_char8_t
#include <filesystem>
#include <fstream>
#include <string>
int main(void) {
  namespace fs = std::filesystem;
  const std::u8string name = u8"unround-probe-日本.txt";
  fs::path p = fs::temp_directory_path() / fs::path(name);
  {
    std::ofstream o(p, std::ios::binary);
    o << "x";
  }
  bool ok = fs::exists(p) && fs::file_size(p) == 1u && p.filename().u8string() == name;
  fs::remove(p);
  return ok ? 0 : 1;
}

//=== probe: parallel_algorithms
//--- title: parallel algorithms (execution::par / par_unseq); prints the threads used
//--- paper: P0024R2
//--- std: c++20
//--- macro: __cpp_lib_execution, __cpp_lib_parallel_algorithm
//--- libs:  | -ltbb
//--- inspect: imports
#include <algorithm>
#include <cstdio>
#include <execution>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>
int main(void) {
  std::vector<double> v(std::size_t{1} << 22);
  std::iota(v.begin(), v.end(), 0.0);
  std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](double& x) -> void { x *= 0.5; });
  std::mutex m;
  std::set<std::thread::id> ids;
  std::for_each(std::execution::par, v.begin(), v.end(), [&](double& x) -> void {
    if (static_cast<long long>(x) % 4096 == 0) {
      std::lock_guard lock(m);
      ids.insert(std::this_thread::get_id());
    }
  });
  double s = std::reduce(std::execution::par_unseq, v.begin(), v.end());
  std::printf("threads=%zu\n", ids.size());
  return s > 0.0 ? 0 : 1;
}

//=== probe: hardware_interference_size
//--- title: std::hardware_destructive_interference_size
//--- paper: P0154R1
//--- std: c++20
//--- macro: __cpp_lib_hardware_interference_size
#include <cstddef>
#include <new>
int main(void) {
  constexpr std::size_t d = std::hardware_destructive_interference_size;
  return d >= std::size_t{32} ? 0 : 1;
}

//=== probe: assume_aligned
//--- title: std::assume_aligned
//--- paper: P1007R3
//--- std: c++20
//--- macro: __cpp_lib_assume_aligned
#include <memory>
int main(void) {
  alignas(64) float buf[16]{};
  float* p = std::assume_aligned<64>(buf);
  p[0] = 1.0f;
  return buf[0] == 1.0f ? 0 : 1;
}

//=== probe: source_location
//--- title: std::source_location
//--- paper: P1208R6
//--- std: c++20
//--- macro: __cpp_lib_source_location
#include <source_location>
int main(void) {
  auto l = std::source_location::current();
  return l.line() > 0u && l.file_name()[0] != '\0' ? 0 : 1;
}

//=== probe: hardening
//--- title: library hardening stops vector[] out of range (vendor switches)
//--- paper: P3471R4 (vendor modes)
//--- flags: -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST -D_GLIBCXX_ASSERTIONS -D_MSVC_STL_HARDENING=1
//--- expect: abnormal
#include <cstddef>
#include <vector>
int main(int argc, char**) {
  std::vector<int> v(2uz);
  volatile std::size_t i = static_cast<std::size_t>(argc) + 5uz;
  volatile int x = v[i];
  static_cast<void>(x);
  return 0;
}
