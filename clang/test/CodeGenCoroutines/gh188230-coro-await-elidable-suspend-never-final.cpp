// Tests that a coro_await_elidable coroutine with a suspend_never final suspend
// destroys a resumed elided callee through the frame destroy slot.
//
// REQUIRES: x86_64-linux
// RUN: %clangxx -std=c++20 -O2 -mllvm -coro-elide-branch-ratio=0 \
// RUN:   -fsanitize=address %s -o %t && %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 \
// RUN:   -mllvm -coro-elide-branch-ratio=0 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct gate {
  std::coroutine_handle<> waiter = nullptr;
  bool open = false;

  struct awaiter {
    gate &g;
    bool await_ready() noexcept { return g.open; }
    void await_suspend(std::coroutine_handle<> h) noexcept { g.waiter = h; }
    void await_resume() noexcept {}
  };

  awaiter operator co_await() noexcept { return {*this}; }

  void set() noexcept {
    open = true;
    if (waiter) {
      auto h = waiter;
      waiter = nullptr;
      h.resume();
    }
  }
};

struct [[clang::coro_await_elidable]] task {
  struct promise_type {
    std::coroutine_handle<> continuation = nullptr;

    task get_return_object() noexcept {
      return {std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }

    void return_void() noexcept {
      if (continuation)
        continuation.resume();
    }

    void unhandled_exception() noexcept { __builtin_abort(); }
  };

  std::coroutine_handle<promise_type> handle;

  bool await_ready() noexcept { return false; }

  void await_suspend(std::coroutine_handle<> h) noexcept {
    handle.promise().continuation = h;
  }

  void await_resume() noexcept {}
};

task callee(gate &g, int &value) {
  co_await g;
  value = 42;
}

task caller(gate &g, int &value, bool &finished) {
  co_await callee(g, value);
  finished = true;
}

int main() {
  gate g;
  int value = 0;
  bool finished = false;

  [[maybe_unused]] auto t = caller(g, value, finished);
  g.set();

  return value == 42 && finished ? 0 : 1;
}

// CHECK-LABEL: define internal void @_Z6calleeR4gateRi.resume(
// CHECK:         %[[DESTROY_ADDR:.+]] = getelementptr inbounds{{.*}} i8, ptr %{{.+}}, i64 8
// CHECK-NEXT:    %[[DESTROY:.+]] = load ptr, ptr %[[DESTROY_ADDR]]
// CHECK:         store i32 42,
// CHECK-NOT:     call void @_Zdl
// CHECK:         tail call void %[[DESTROY]](
// CHECK-NEXT:    ret void

// CHECK-LABEL: define internal {{.*}}void @_Z6calleeR4gateRi.destroy(
// CHECK:         call void @_Zdl

// CHECK-LABEL: define internal {{.*}}void @_Z6calleeR4gateRi.cleanup(
// CHECK-NOT:     call void @_Zdl
// CHECK:         ret void
