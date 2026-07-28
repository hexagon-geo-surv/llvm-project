// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// CHECK-LABEL: @length
// CHECK-SAME:      %[[INPUT:.*]]: !list.list<i32>
//       CHECK:   %[[C1:.*]] = arith.constant 1 : i32
//       CHECK:   %[[TRUE:.*]] = arith.constant true
//       CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:   %[[LOOP:.*]]:2 = scf.while (%[[ARG:.*]] = %[[INPUT]], %[[COUNT:.*]] = %[[C0]]) : (!list.list<i32>, i32) -> (!list.list<i32>, i32) {
//       CHECK:     %[[IS_EMPTY:.*]] = list.is_empty %[[ARG]] : !list.list<i32> -> i1
//       CHECK:     %[[COND:.*]] = arith.xori %[[IS_EMPTY]], %[[TRUE]] : i1
//       CHECK:     scf.condition(%[[COND]]) %[[ARG]], %[[COUNT]] : !list.list<i32>, i32
//       CHECK:   } do {
//       CHECK:   ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[COUNT:.*]]: i32):
//       CHECK:     %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
//       CHECK:     %[[NEXT:.*]] = arith.addi %[[COUNT]], %[[C1]] : i32
//       CHECK:     scf.yield %[[REST]], %[[NEXT]] : !list.list<i32>, i32
//       CHECK:   }
//       CHECK:   return %[[LOOP]]#1 : i32
func.func @length(%input: !list.list<i32>) -> i32 {
  %length = list.length %input : !list.list<i32> -> i32
  return %length : i32
}

// -----

// The elements of the list are of no interest, only how many there are.

// CHECK-LABEL: @length_of_wide_elements
//       CHECK:   scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (!list.list<i64>, i32) -> (!list.list<i64>, i32)
//       CHECK:     list.is_empty %{{.*}} : !list.list<i64> -> i1
//       CHECK:   } do {
//       CHECK:     list.pop_front %{{.*}} : !list.list<i64>
//       CHECK:     arith.addi
func.func @length_of_wide_elements(%input: !list.list<i64>) -> i32 {
  %length = list.length %input : !list.list<i64> -> i32
  return %length : i32
}

// -----

// Nothing is known about the length of a list that an element is appended to,
// so its elements are counted like those of any other list. (The length of a
// list built by a `list.map` is read through the map instead, which the
// simplification tests cover.)

// CHECK-LABEL: @length_of_pushed
// CHECK-SAME:      %[[INPUT:[^:]*]]: !list.list<i32>, %[[ITEM:[^:]*]]: i32
//       CHECK:   %[[LONGER:.*]] = list.push_back %[[INPUT]], %[[ITEM]]
//       CHECK:   %[[LOOP:.*]]:2 = scf.while (%{{.*}} = %[[LONGER]],
//       CHECK:   } do {
//       CHECK:     list.pop_front
//       CHECK:     arith.addi
//       CHECK:   }
//       CHECK:   return %[[LOOP]]#1 : i32
func.func @length_of_pushed(%input: !list.list<i32>, %item: i32) -> i32 {
  %longer = list.push_back %input, %item : !list.list<i32>
  %length = list.length %longer : !list.list<i32> -> i32
  return %length : i32
}
