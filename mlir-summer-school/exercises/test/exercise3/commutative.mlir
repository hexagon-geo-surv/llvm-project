// RUN: school-opt %s -canonicalize | FileCheck %s

// Exercise 3 stretch (a): the Commutative trait. Trait folding moves constant
// operands of commutative ops to the right -- for free, with no code written.
// This is why upstream folders only ever check the rhs for constants: after
// canonicalization, that is where constants live.

// CHECK-LABEL: func.func @constant_moves_right
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @constant_moves_right(%x: i32) -> i32 {
  // CHECK: %[[C5:.+]] = arith.constant 5 : i32
  %c5 = arith.constant 5 : i32
  // CHECK: %[[R:.+]] = school.max %[[X]], %[[C5]] : i32
  %r = school.max %c5, %x : i32
  // CHECK: return %[[R]]
  return %r : i32
}
