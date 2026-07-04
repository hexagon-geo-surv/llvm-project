// RUN: school-opt %s -canonicalize | FileCheck %s

// Exercise 3 stretch (b): a real canonicalization pattern, hooked into
// -canonicalize via hasCanonicalizer. max(max(x, c1), c2) needs a *new*
// constant op for max(c1, c2), so it cannot be a fold -- it must be a
// RewritePattern. The pattern only checks the rhs for constants: the
// Commutative trait (stretch (a)) already established that normal form.

// CHECK-LABEL: func.func @reassociate
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @reassociate(%x: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %c7 = arith.constant 7 : i32
  // CHECK: %[[C7:.+]] = arith.constant 7 : i32
  %a = school.max %x, %c3 : i32
  // CHECK: %[[R:.+]] = school.max %[[X]], %[[C7]] : i32
  // CHECK-NOT: school.max
  %b = school.max %a, %c7 : i32
  // CHECK: return %[[R]]
  return %b : i32
}

// Constants on the *left* still work: the Commutative trait normalizes them
// to the right before the pattern runs (patterns + trait folds compose in
// the same greedy driver).
// CHECK-LABEL: func.func @reassociate_constants_left
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @reassociate_constants_left(%x: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %c9 = arith.constant 9 : i32
  // CHECK: %[[C9:.+]] = arith.constant 9 : i32
  %a = school.max %c4, %x : i32
  // CHECK: %[[R:.+]] = school.max %[[X]], %[[C9]] : i32
  // CHECK-NOT: school.max
  %b = school.max %c9, %a : i32
  // CHECK: return %[[R]]
  return %b : i32
}
