// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Feature probes: the core language of C++23 and C++26. Each probe below is
// compiled, linked and run on its own by run.py, and passes when it exits
// with status 0. The probes follow the project's rules for C++: an empty
// parameter list is written (void), and no conversion is left implicit.

//=== probe: deducing_this
//--- title: explicit object parameter (deducing this)
//--- paper: P0847R7
//--- macro: __cpp_explicit_this_parameter
struct S {
  int v = 1;
  template <class Self> auto&& get(this Self&& self) { return static_cast<Self&&>(self).v; }
  int fact(this S const& self, int n) { return n <= 1 ? 1 : n * self.fact(n - 1); }
};
int main(void) {
  S s;
  S const cs{2};
  s.get() = 5;
  return s.get() == 5 && cs.get() == 2 && s.fact(5) == 120 ? 0 : 1;
}

//=== probe: multidimensional_subscript
//--- title: multidimensional subscript operator a[i, j]
//--- paper: P2128R6
//--- macro: __cpp_multidimensional_subscript
struct Grid {
  int d[6]{};
  constexpr int& operator[](int i, int j) { return d[i * 3 + j]; }
};
int main(void) {
  Grid g;
  g[1, 2] = 7;
  return g.d[5] == 7 ? 0 : 1;
}

//=== probe: if_consteval
//--- title: if consteval
//--- paper: P1938R3
//--- macro: __cpp_if_consteval
constexpr int twice_or_thrice(int x) {
  if consteval {
    return 2 * x;
  } else {
    return 3 * x;
  }
}
static_assert(twice_or_thrice(2) == 4);
int main(int argc, char**) { return twice_or_thrice(argc) == 3 * argc ? 0 : 1; }

//=== probe: static_call_operator
//--- title: static operator() and static lambdas
//--- paper: P1169R4
//--- macro: __cpp_static_call_operator
struct Add {
  static int operator()(int a, int b) { return a + b; }
};
int main(void) {
  auto inc = [](int x) static { return x + 1; };
  return Add{}(1, 2) == 3 && Add::operator()(2, 3) == 5 && inc(1) == 2 ? 0 : 1;
}

//=== probe: auto_cast
//--- title: auto(x) decay-copy
//--- paper: P0849R8
//--- macro: __cpp_auto_cast
#include <vector>
int main(void) {
  std::vector<int> v{1, 2, 3};
  auto w = auto(v);
  w[0] = 9;
  return v[0] == 1 && w[0] == 9 ? 0 : 1;
}

//=== probe: assume_attribute
//--- title: [[assume(expr)]]
//--- paper: P1774R8
//--- macro: __has_cpp_attribute(assume)
#if !__has_cpp_attribute(assume)
#error "[[assume]] is not supported"
#endif
static int half(int x) {
  [[assume(x >= 0)]];
  return x / 2;
}
int main(int argc, char**) { return half(argc * 2) == argc ? 0 : 1; }

//=== probe: size_t_literal
//--- title: uz / z integer literal suffixes
//--- paper: P0330R8
//--- macro: __cpp_size_t_suffix
#include <cstddef>
#include <type_traits>
int main(void) {
  auto a = 42uz;
  auto b = 42z;
  static_assert(std::is_same_v<decltype(a), std::size_t>);
  static_assert(std::is_signed_v<decltype(b)>);
  return a == 42uz && b == 42z ? 0 : 1;
}

//=== probe: constexpr_static_local
//--- title: static constexpr variables in constexpr functions
//--- paper: P2647R1
//--- macro: __cpp_constexpr
constexpr char digit(int i) {
  static constexpr char table[] = "0123456789";
  return table[i];
}
static_assert(digit(3) == '3');
int main(void) { return digit(9) == '9' ? 0 : 1; }

//=== probe: elifdef
//--- title: #elifdef / #elifndef
//--- paper: P2334R1
#define PROBE_A
#ifdef PROBE_NOT_DEFINED
#error "wrong branch"
#elifdef PROBE_A
constexpr bool ok = true;
#else
#error "wrong branch"
#endif
int main(void) { return ok ? 0 : 1; }

//=== probe: consteval_escalation
//--- title: consteval propagates up (immediate-escalating functions)
//--- paper: P2564R3
//--- macro: __cpp_consteval
consteval int id(int x) { return x; }
template <class T> constexpr int call(T t) { return id(t); }
static_assert(call(3) == 3);
int main(void) { return 0; }

//=== probe: float16_t
//--- title: std::float16_t (<stdfloat>)
//--- paper: P1467R9
//--- macro: __STDCPP_FLOAT16_T__
#include <stdfloat>
#if !defined(__STDCPP_FLOAT16_T__)
#error "std::float16_t is not provided"
#endif
int main(void) {
  std::float16_t h = 1.5f16;
  float f = static_cast<float>(h) * 2.0f;
  return f == 3.0f ? 0 : 1;
}

//=== probe: bfloat16_t
//--- title: std::bfloat16_t (<stdfloat>)
//--- paper: P1467R9
//--- macro: __STDCPP_BFLOAT16_T__
#include <stdfloat>
#if !defined(__STDCPP_BFLOAT16_T__)
#error "std::bfloat16_t is not provided"
#endif
int main(void) {
  std::bfloat16_t b = 1.5bf16;
  return static_cast<float>(b) == 1.5f ? 0 : 1;
}

//=== probe: builtin_overflow
//--- title: __builtin_mul_overflow / __builtin_add_overflow (checked sizes)
//--- paper: Clang builtin
#include <cstddef>
int main(void) {
  std::size_t bytes = 0uz;
  bool overflow = __builtin_mul_overflow(std::size_t{1} << 40, std::size_t{1} << 40, &bytes);
  int sum = 0;
  bool fine = !__builtin_add_overflow(2, 3, &sum);
  return overflow && fine && sum == 5 ? 0 : 1;
}

//=== probe: pack_indexing
//--- title: pack indexing Ts...[I]
//--- paper: P2662R3
//--- std: c++26
//--- macro: __cpp_pack_indexing
#include <type_traits>
template <class... Ts> using First = Ts...[0];
template <auto... Vs> constexpr auto last = Vs...[sizeof...(Vs) - 1];
static_assert(std::is_same_v<First<int, char>, int>);
static_assert(last<1, 2, 3> == 3);
int main(void) { return 0; }

//=== probe: deleted_with_reason
//--- title: = delete("reason")
//--- paper: P2573R2
//--- std: c++26
//--- macro: __cpp_deleted_function
static void f(int) {}
static void f(double) = delete("use the int overload");
int main(void) {
  f(1);
  return 0;
}

//=== probe: placeholder_variables
//--- title: placeholder variables named _
//--- paper: P2169R4
//--- std: c++26
//--- macro: __cpp_placeholder_variables
#include <utility>
int main(void) {
  auto _ = 1;
  auto _ = 2.0;
  auto [a, _] = std::pair{3, 4};
  return a == 3 ? 0 : 1;
}

//=== probe: structured_binding_pack
//--- title: structured bindings that introduce a pack
//--- paper: P1061R10
//--- std: c++26
//--- macro: __cpp_structured_bindings
#include <tuple>
template <class T> constexpr int sum(T t) {
  auto [... xs] = t;
  return (0 + ... + xs);
}
int main(void) { return sum(std::tuple{1, 2, 3}) == 6 ? 0 : 1; }

//=== probe: structured_binding_condition
//--- title: structured binding declaration as a condition
//--- paper: P0963R3
//--- std: c++26
struct Result {
  int value;
  int error;
  explicit operator bool(void) const { return error == 0; }
};
static Result compute(int x) { return {x * 2, 0}; }
int main(void) {
  if (auto [value, error] = compute(21)) return value == 42 && error == 0 ? 0 : 1;
  return 1;
}

//=== probe: constexpr_placement_new
//--- title: placement new in constant expressions
//--- paper: P2747R2
//--- std: c++26
#include <memory>
#include <new>
struct Box {
  int v;
};
constexpr int make(void) {
  std::allocator<Box> a;
  Box* p = a.allocate(1uz);
  ::new (static_cast<void*>(p)) Box{42};
  int r = p->v;
  std::destroy_at(p);
  a.deallocate(p, 1uz);
  return r;
}
static_assert(make() == 42);
int main(void) { return 0; }

//=== probe: constexpr_void_cast
//--- title: cast from void* in constant expressions
//--- paper: P2738R1
//--- std: c++26
constexpr int through_void(void) {
  int x = 5;
  void* p = &x;
  return *static_cast<int*>(p);
}
static_assert(through_void() == 5);
int main(void) { return 0; }

//=== probe: static_assert_message
//--- title: user-generated static_assert messages
//--- paper: P2741R3
//--- std: c++26
//--- macro: __cpp_static_assert
#include <cstddef>
struct Message {
  constexpr std::size_t size(void) const { return 2uz; }
  constexpr const char* data(void) const { return "ok"; }
};
static_assert(true, Message{});
int main(void) { return 0; }

//=== probe: embed_cxx
//--- title: #embed in C++
//--- paper: P1967R14
//--- std: c++26
//--- macro: __cpp_pp_embed
static constexpr unsigned char self[] = {
#embed __FILE__
};
int main(void) { return sizeof(self) > 100uz && self[0] == static_cast<unsigned char>('/') ? 0 : 1; }

//=== probe: contracts
//--- title: contracts: pre, post, contract_assert
//--- paper: P2900R14
//--- std: c++26
//--- macro: __cpp_contracts
//--- flags:  | -fcontracts | -fexperimental-contracts
static int half(int x) pre(x >= 0) post(r : r * 2 <= x) {
  contract_assert(x < 1000);
  return x / 2;
}
int main(void) { return half(10) == 5 ? 0 : 1; }

//=== probe: reflection_core
//--- title: the reflection operator ^^ (compiler side of P2996, without <meta>)
//--- paper: P2996R13
//--- std: c++26
//--- macro: __cpp_impl_reflection
//--- flags:  | -freflection | -freflection-latest
constexpr auto reflected = ^^int;
int main(void) { return 0; }

//=== probe: reflection
//--- title: static reflection with std::meta
//--- paper: P2996R13
//--- std: c++26
//--- macro: __cpp_impl_reflection, __cpp_lib_reflection
//--- flags:  | -freflection | -freflection-latest
#include <meta>
struct Options {
  int max_iter;
  float tol;
};
int main(void) {
  constexpr auto ctx = std::meta::access_context::unchecked();
  constexpr auto count = std::meta::nonstatic_data_members_of(^^Options, ctx).size();
  return count == 2uz ? 0 : 1;
}

//=== probe: expansion_statements
//--- title: expansion statements (template for)
//--- paper: P1306R5
//--- std: c++26
//--- macro: __cpp_expansion_statements
#include <tuple>
int main(void) {
  int sum = 0;
  template for (auto x : std::tuple{1, 2L, 3u}) sum += static_cast<int>(x);
  return sum == 6 ? 0 : 1;
}

//=== probe: trivial_unions
//--- title: unions of non-trivial members are trivially constructible
//--- paper: P3074R7
//--- std: c++26
//--- macro: __cpp_trivial_union
#include <memory>
#include <string>
union Slot {
  std::string s;  // before C++26 the union's default constructor and destructor are deleted
};
int main(void) {
  Slot slot;
  std::construct_at(&slot.s, "x");
  bool ok = slot.s == "x";
  std::destroy_at(&slot.s);
  return ok ? 0 : 1;
}
