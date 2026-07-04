// RUN: school-opt %s -pass-pipeline="builtin.module(func.func(school-strength-reduce))" | FileCheck %s

// Exercise 1 checkpoints 2+3: muli-by-power-of-two becomes shli; everything
// else is left alone; several rewrites in one function must all happen
// (mutating while walking is the classic bug this flushes out).
//
// Note: the hand-written pass does NOT clean up the now-dead constants --
// that is deliberate (Session 3 shows who cleans them up).

// CHECK-LABEL: func.func @mul_by_8
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @mul_by_8(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  // CHECK: %[[C3:.+]] = arith.constant 3 : i32
  // CHECK: %[[R:.+]] = arith.shli %[[X]], %[[C3]] : i32
  %r = arith.muli %x, %c8 : i32
  // CHECK-NOT: arith.muli
  // CHECK: return %[[R]]
  return %r : i32
}

// A constant that is not a power of two must be left untouched.
// CHECK-LABEL: func.func @mul_by_7
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @mul_by_7(%x: i32) -> i32 {
  // CHECK: %[[C7:.+]] = arith.constant 7 : i32
  %c7 = arith.constant 7 : i32
  // CHECK-NOT: arith.shli
  // CHECK: %[[R:.+]] = arith.muli %[[X]], %[[C7]] : i32
  %r = arith.muli %x, %c7 : i32
  // CHECK: return %[[R]]
  return %r : i32
}

// A non-constant rhs must be left untouched (and must not crash the pass).
// CHECK-LABEL: func.func @mul_by_dynamic
// CHECK-SAME:  (%[[X:.+]]: i32, %[[Y:.+]]: i32)
func.func @mul_by_dynamic(%x: i32, %y: i32) -> i32 {
  // CHECK-NOT: arith.shli
  // CHECK: %[[R:.+]] = arith.muli %[[X]], %[[Y]] : i32
  %r = arith.muli %x, %y : i32
  // CHECK: return %[[R]]
  return %r : i32
}

// Two independent rewrites in one function: both must happen.
// CHECK-LABEL: func.func @two_rewrites
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @two_rewrites(%x: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  // CHECK: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK: %[[S1:.+]] = arith.shli %[[X]], %[[C2]] : i32
  %a = arith.muli %x, %c4 : i32
  // CHECK: %[[C4A:.+]] = arith.constant 4 : i32
  // CHECK: %[[S2:.+]] = arith.shli %[[S1]], %[[C4A]] : i32
  %b = arith.muli %a, %c16 : i32
  // CHECK-NOT: arith.muli
  // CHECK: return %[[S2]]
  return %b : i32
}
