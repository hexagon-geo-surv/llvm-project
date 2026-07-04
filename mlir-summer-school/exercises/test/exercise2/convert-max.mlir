// RUN: school-opt %s -pass-pipeline="builtin.module(convert-school-to-arith)" | FileCheck %s

// Exercise 2B checkpoint 1: school.max is lowered to arith.cmpi + arith.select.
// This is a *partial* conversion: ops the target says nothing about (func.func,
// func.return) survive untouched -- only `school` ops must disappear.

// CHECK-LABEL: func.func @lower_max
// CHECK-SAME:  (%[[A:.+]]: i32, %[[B:.+]]: i32)
func.func @lower_max(%a: i32, %b: i32) -> i32 {
  // CHECK: %[[CMP:.+]] = arith.cmpi sgt, %[[A]], %[[B]] : i32
  // CHECK: %[[SEL:.+]] = arith.select %[[CMP]], %[[A]], %[[B]] : i32
  %m = school.max %a, %b : i32
  // CHECK-NOT: school.
  // CHECK: return %[[SEL]]
  return %m : i32
}

// A chain of maxes: the second max must consume the lowered value of the
// first (this is what the adaptor gives you -- the already-converted
// operands).
// CHECK-LABEL: func.func @lower_max_chain
// CHECK-SAME:  (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func.func @lower_max_chain(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: %[[CMP1:.+]] = arith.cmpi sgt, %[[A]], %[[B]] : i32
  // CHECK: %[[SEL1:.+]] = arith.select %[[CMP1]], %[[A]], %[[B]] : i32
  %m1 = school.max %a, %b : i32
  // CHECK: %[[CMP2:.+]] = arith.cmpi sgt, %[[SEL1]], %[[C]] : i32
  // CHECK: %[[SEL2:.+]] = arith.select %[[CMP2]], %[[SEL1]], %[[C]] : i32
  %m2 = school.max %m1, %c : i32
  // CHECK-NOT: school.
  // CHECK: return %[[SEL2]]
  return %m2 : i32
}
