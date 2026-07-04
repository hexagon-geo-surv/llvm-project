// RUN: school-opt %s -pass-pipeline="builtin.module(convert-school-to-arith)" | FileCheck %s

// Exercise 2B checkpoints 2+3. Before you write the MacOpLowering pattern,
// run this input manually and read the error: the school dialect is marked
// illegal, so a partial conversion *fails loudly* on a school.mac it has no
// pattern for ("failed to legalize operation 'school.mac' that was
// explicitly marked illegal"). After the pattern exists, this test is green.

// CHECK-LABEL: func.func @lower_mac
// CHECK-SAME:  (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func.func @lower_mac(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: %[[MUL:.+]] = arith.muli %[[A]], %[[B]] : i32
  // CHECK: %[[ADD:.+]] = arith.addi %[[MUL]], %[[C]] : i32
  %r = school.mac %a, %b, %c : i32
  // CHECK-NOT: school.
  // CHECK: return %[[ADD]]
  return %r : i32
}

// max and mac together: both lowerings must fire in the same run.
// CHECK-LABEL: func.func @lower_mixed
// CHECK-SAME:  (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func.func @lower_mixed(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: %[[MUL:.+]] = arith.muli %[[A]], %[[B]] : i32
  // CHECK: %[[ADD:.+]] = arith.addi %[[MUL]], %[[C]] : i32
  %m = school.mac %a, %b, %c : i32
  // CHECK: %[[CMP:.+]] = arith.cmpi sgt, %[[ADD]], %[[C]] : i32
  // CHECK: %[[SEL:.+]] = arith.select %[[CMP]], %[[ADD]], %[[C]] : i32
  %r = school.max %m, %c : i32
  // CHECK-NOT: school.
  // CHECK: return %[[SEL]]
  return %r : i32
}
