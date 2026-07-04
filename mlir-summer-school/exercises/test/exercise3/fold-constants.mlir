// RUN: school-opt %s -canonicalize | FileCheck %s

// Exercise 3 checkpoint 3: constant folding. MaxOp::fold returns an
// *Attribute*; someone must turn that attribute back into an op. That someone
// is the dialect's constant materializer -- without it, the fold result is
// silently dropped and this test stays red even though the fold "works".

// CHECK-LABEL: func.func @max_of_constants
func.func @max_of_constants() -> i32 {
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  // CHECK: %[[C5:.+]] = arith.constant 5 : i32
  // CHECK-NOT: school.max
  %r = school.max %c3, %c5 : i32
  // CHECK: return %[[C5]]
  return %r : i32
}

// Signed comparison: -7 < 2, so the max is 2 (an unsigned max would pick -7).
// CHECK-LABEL: func.func @max_of_negative
func.func @max_of_negative() -> i32 {
  %cm7 = arith.constant -7 : i32
  %c2 = arith.constant 2 : i32
  // CHECK: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-NOT: school.max
  %r = school.max %cm7, %c2 : i32
  // CHECK: return %[[C2]]
  return %r : i32
}
