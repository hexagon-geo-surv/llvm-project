// RUN: school-opt %s -canonicalize | FileCheck %s

// Exercise 3 checkpoint 2: max(x, x) == x is a fold -- it replaces the op
// with an *existing* value, creating nothing. -canonicalize (the greedy
// driver) invokes MaxOp::fold; no school-specific pass is involved.

// CHECK-LABEL: func.func @max_same
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @max_same(%x: i32) -> i32 {
  // CHECK-NOT: school.max
  %r = school.max %x, %x : i32
  // CHECK: return %[[X]]
  return %r : i32
}

// The fold must not fire when the operands differ.
// CHECK-LABEL: func.func @max_different
// CHECK-SAME:  (%[[X:.+]]: i32, %[[Y:.+]]: i32)
func.func @max_different(%x: i32, %y: i32) -> i32 {
  // CHECK: %[[R:.+]] = school.max %[[X]], %[[Y]] : i32
  %r = school.max %x, %y : i32
  // CHECK: return %[[R]]
  return %r : i32
}
