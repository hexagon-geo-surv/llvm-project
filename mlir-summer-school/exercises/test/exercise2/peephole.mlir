// RUN: school-opt %s -pass-pipeline="builtin.module(func.func(school-peephole))" | FileCheck %s

// Exercise 2A: two patterns run to a fixpoint by the greedy driver.
// The @mul_chain case is the payoff: neither pattern alone reduces
// ((x*4)*8) to a single shift -- their *composition* under the fixpoint
// driver does. Note that dead constants vanish too: the greedy driver
// performs DCE, unlike the hand-written pass from Exercise 1.

// Checkpoint 1: pattern MulByPow2ToShl fires on a single muli.
// CHECK-LABEL: func.func @mul_by_16
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @mul_by_16(%x: i32) -> i32 {
  %c16 = arith.constant 16 : i32
  // CHECK: %[[C4:.+]] = arith.constant 4 : i32
  // CHECK: %[[R:.+]] = arith.shli %[[X]], %[[C4]] : i32
  %r = arith.muli %x, %c16 : i32
  // CHECK-NOT: arith.muli
  // CHECK: return %[[R]]
  return %r : i32
}

// Checkpoint 2: pattern MergeConsecutiveShl fires on a shift-of-shift.
// CHECK-LABEL: func.func @shl_shl
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @shl_shl(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %a = arith.shli %x, %c1 : i32
  // CHECK: %[[C3:.+]] = arith.constant 3 : i32
  // CHECK: %[[R:.+]] = arith.shli %[[X]], %[[C3]] : i32
  %b = arith.shli %a, %c2 : i32
  // CHECK: return %[[R]]
  return %b : i32
}

// Checkpoint 3 (composition): ((x*4)*8) collapses to one shli by 5. This
// needs MulByPow2ToShl twice, then MergeConsecutiveShl on the result --
// i.e. the driver must revisit IR produced by earlier rewrites.
// CHECK-LABEL: func.func @mul_chain
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @mul_chain(%x: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %a = arith.muli %x, %c4 : i32
  // CHECK: %[[C5:.+]] = arith.constant 5 : i32
  // CHECK: %[[R:.+]] = arith.shli %[[X]], %[[C5]] : i32
  %b = arith.muli %a, %c8 : i32
  // CHECK-NOT: arith.muli
  // CHECK-NOT: arith.shli
  // CHECK: return %[[R]]
  return %b : i32
}

// Overflow guard: 20 + 15 >= 32, so merging would shift past the bit width
// (poison). The chain must be left alone.
// CHECK-LABEL: func.func @no_merge_overflow
// CHECK-SAME:  (%[[X:.+]]: i32)
func.func @no_merge_overflow(%x: i32) -> i32 {
  // CHECK-DAG: %[[C20:.+]] = arith.constant 20 : i32
  // CHECK-DAG: %[[C15:.+]] = arith.constant 15 : i32
  %c20 = arith.constant 20 : i32
  %c15 = arith.constant 15 : i32
  // CHECK: %[[A:.+]] = arith.shli %[[X]], %[[C20]] : i32
  %a = arith.shli %x, %c20 : i32
  // CHECK: %[[B:.+]] = arith.shli %[[A]], %[[C15]] : i32
  %b = arith.shli %a, %c15 : i32
  // CHECK: return %[[B]]
  return %b : i32
}
