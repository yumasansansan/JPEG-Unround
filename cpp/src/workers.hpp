// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Threads that take the parts of a task at once: the caller, which takes part 0, and
// count - 1 threads that wait between tasks. A part may wait for every other at a
// barrier (meet), as often as every other part does. Where a thread cannot be made,
// the constructor throws what making it threw (std::system_error), once the threads
// made are joined.
#ifndef UNROUND_WORKERS_HPP
#define UNROUND_WORKERS_HPP

#include <atomic>
#include <barrier>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace unround::pdhg::detail {

class Workers {
 public:
  explicit Workers(std::size_t count)
      : count_(count == 0u ? 1u : count),
        start_(static_cast<std::ptrdiff_t>(count_)),
        done_(static_cast<std::ptrdiff_t>(count_)),
        meeting_(static_cast<std::ptrdiff_t>(count_)) {
    threads_.reserve(count_ - 1u);
    try {
      for (std::size_t part = 1; part < count_; ++part) {
        threads_.emplace_back([this, part](void) -> void { serve(part); });
      }
    } catch (...) {
      // A thread that could not be made: the threads made are let go at their start,
      // where the caller and the threads not made arrive at once, and joined.
      stopping_.store(true, std::memory_order_relaxed);
      static_cast<void>(start_.arrive(static_cast<std::ptrdiff_t>(count_ - threads_.size())));
      threads_.clear();
      throw;
    }
  }

  Workers(const Workers&) = delete;
  Workers& operator=(const Workers&) = delete;
  Workers(Workers&&) = delete;
  Workers& operator=(Workers&&) = delete;

  // The threads stop at their next start, and are joined.
  ~Workers(void) {
    stopping_.store(true, std::memory_order_relaxed);
    start_.arrive_and_wait();
  }

  // Runs task(part) for every part at once, part 0 on the caller, and returns once every
  // part has. A task does not throw: one that did would end the program.
  void run(const std::function<void(std::size_t)>& task) noexcept {
    task_ = &task;
    start_.arrive_and_wait();
    task(0);
    done_.arrive_and_wait();
  }

  // Waits until every part of the task has come here.
  void meet(void) noexcept {
    if (count_ > 1u) meeting_.arrive_and_wait();
  }

 private:
  void serve(std::size_t part) noexcept {
    for (;;) {
      start_.arrive_and_wait();
      if (stopping_.load(std::memory_order_relaxed)) return;
      (*task_)(part);
      done_.arrive_and_wait();
    }
  }

  std::size_t count_;
  std::barrier<> start_;
  std::barrier<> done_;
  std::barrier<> meeting_;
  std::atomic<bool> stopping_{false};
  const std::function<void(std::size_t)>* task_ = nullptr;
  // Last, so that the threads are joined before the barriers go.
  std::vector<std::jthread> threads_;
};

}  // namespace unround::pdhg::detail

#endif  // UNROUND_WORKERS_HPP
