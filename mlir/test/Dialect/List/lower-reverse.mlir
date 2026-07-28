// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// CHECK-LABEL: @reverse
// CHECK-SAME:      %[[INPUT:.*]]: !list.list<i32>
//       CHECK:   %[[TRUE:.*]] = arith.constant true
//       CHECK:   %[[EMPTY:.*]] = list.empty : !list.list<i32>
//       CHECK:   %[[LOOP:.*]]:2 = scf.while (%[[ARG:.*]] = %[[INPUT]], %[[ACC:.*]] = %[[EMPTY]]) : (!list.list<i32>, !list.list<i32>) -> (!list.list<i32>, !list.list<i32>) {
//       CHECK:     %[[IS_EMPTY:.*]] = list.is_empty %[[ARG]] : !list.list<i32> -> i1
//       CHECK:     %[[COND:.*]] = arith.xori %[[IS_EMPTY]], %[[TRUE]] : i1
//       CHECK:     scf.condition(%[[COND]]) %[[ARG]], %[[ACC]] : !list.list<i32>, !list.list<i32>
//       CHECK:   } do {
//       CHECK:   ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i32>):
//       CHECK:     %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
//       CHECK:     %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
// The element that is moved first ends up at the back of the result list.
//       CHECK:     %[[LONGER:.*]] = list.push_front %[[ACC]], %[[ELEM]] : !list.list<i32>
//       CHECK:     scf.yield %[[REST]], %[[LONGER]] : !list.list<i32>, !list.list<i32>
//       CHECK:   }
//       CHECK:   return %[[LOOP]]#1 : !list.list<i32>
func.func @reverse(%input: !list.list<i32>) -> !list.list<i32> {
  %reversed = list.reverse %input : !list.list<i32>
  return %reversed : !list.list<i32>
}

// -----

// There is nothing to move, so no loop is built. The pattern has a higher
// benefit than the lowering, so it is applied first.

// CHECK-LABEL: @reverse_empty
// The list that was reversed has no users left: this pass does not remove dead
// code.
//       CHECK:   list.empty : !list.list<i32>
//       CHECK:   %[[EMPTY:.*]] = list.empty : !list.list<i32>
//       CHECK:   return %[[EMPTY]] : !list.list<i32>
//   CHECK-NOT:   scf.while
func.func @reverse_empty() -> !list.list<i32> {
  %empty = list.empty : !list.list<i32>
  %reversed = list.reverse %empty : !list.list<i32>
  return %reversed : !list.list<i32>
}

// -----

// Nothing is known about the operand list, so a loop is built for it.

// CHECK-LABEL: @reverse_twice
//       CHECK:   %[[FIRST:.*]]:2 = scf.while
//       CHECK:     list.push_front
//       CHECK:   }
//       CHECK:   %[[SECOND:.*]]:2 = scf.while (%{{.*}} = %[[FIRST]]#1,
//       CHECK:     list.push_front
//       CHECK:   }
//       CHECK:   return %[[SECOND]]#1
func.func @reverse_twice(%input: !list.list<i32>) -> !list.list<i32> {
  %once = list.reverse %input : !list.list<i32>
  %twice = list.reverse %once : !list.list<i32>
  return %twice : !list.list<i32>
}
