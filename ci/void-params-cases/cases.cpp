// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cases for ci/void-params.py --self-test: every line that ends in the comment
// "want: void" has to be reported, and no other line. The file compiles as C++26 as it is.

#include <functional>
#include <memory>
#include <string>
#include <vector>

int f();  // want: void
int f(void) { return 1; }

struct Widget {
  Widget();                         // want: void
  explicit Widget(int value) : value_{value} {}
  ~Widget();                        // want: void
  int get() const;                  // want: void
  int operator()() const;           // want: void
  explicit operator bool() const;   // want: void
  int proper(void) const { return value_; }
  int value_ = 0;
};

Widget::Widget() = default;                          // want: void
Widget::~Widget(void) = default;
int Widget::get() const { return value_; }           // want: void
int Widget::operator()(void) const { return value_; }
Widget::operator bool(void) const { return value_ != 0; }

std::function<int()> callback;          // want: void
std::function<int(void)> proper_callback;
using Handler = void (*)();             // want: void
using ProperHandler = void (*)(void);

template <typename T>
T make() {  // want: void
  return T{};
}

auto with_list = [](void) { return 1; };
auto without_list = [] { return 2; };                   // want: void
auto empty_list = []() { return 3; };                   // want: void
auto with_capture = [value = 4]() mutable { return value++; };  // want: void
auto bare_mutable = [value = 5] mutable { return value++; };    // want: void

#if defined(NOT_DEFINED_ANYWHERE)
int inactive();
#endif

int calls(void) {
  Widget widget;
  Widget other{1};
  std::vector<int> numbers(3);
  std::string text = std::string{};
  auto pointer = std::make_unique<Widget>();
  int sum = f() + widget.get() + widget() + widget.operator()() + pointer->get();
  sum += with_list() + without_list() + empty_list() + with_capture() + bare_mutable();
  sum += make<int>() + static_cast<int>(text.size()) + numbers[0];
  sum += static_cast<int>(static_cast<bool>(other)) + (callback ? callback() : 0);
  sum += decltype(f()){};
  int value = int();
  Widget vexing();  // want: void
  return sum + value + numbers.at(1) + [](void) { return 0; }();
}

int main(void) { return calls() == -1; }
