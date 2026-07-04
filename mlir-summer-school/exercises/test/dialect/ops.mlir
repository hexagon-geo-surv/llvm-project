// RUN: school-opt %s | FileCheck %s

// Checkpoint 0 (green from day one): the school ops parse, verify, and print
// back in their custom syntax. If this test fails, the dialect itself is
// broken -- fix that before attempting any exercise.

// CHECK-LABEL: func.func @max
// CHECK-SAME:  (%[[A:.+]]: i32, %[[B:.+]]: i32)
func.func @max(%a: i32, %b: i32) -> i32 {
  // CHECK: %[[M:.+]] = school.max %[[A]], %[[B]] : i32
  %m = school.max %a, %b : i32
  // CHECK: return %[[M]]
  return %m : i32
}

// CHECK-LABEL: func.func @mac
// CHECK-SAME:  (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func.func @mac(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: %[[R:.+]] = school.mac %[[A]], %[[B]], %[[C]] : i32
  %r = school.mac %a, %b, %c : i32
  // CHECK: return %[[R]]
  return %r : i32
}
