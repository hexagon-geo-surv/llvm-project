// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// CHECK-LABEL: @print
// CHECK-SAME:      %[[INPUT:.*]]: !list.list<i32>
//       CHECK:   %[[TRUE:.*]] = arith.constant true
//       CHECK:   scf.while (%[[ARG:.*]] = %[[INPUT]]) : (!list.list<i32>) -> !list.list<i32> {
//       CHECK:     %[[IS_EMPTY:.*]] = list.is_empty %[[ARG]] : !list.list<i32> -> i1
//       CHECK:     %[[COND:.*]] = arith.xori %[[IS_EMPTY]], %[[TRUE]] : i1
//       CHECK:     scf.condition(%[[COND]]) %[[ARG]] : !list.list<i32>
//       CHECK:   } do {
//       CHECK:   ^bb0(%[[ARG:.*]]: !list.list<i32>):
//       CHECK:     %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
//       CHECK:     vector.print %[[ELEM]] : i32
//       CHECK:     %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
//       CHECK:     scf.yield %[[REST]] : !list.list<i32>
//       CHECK:   }
//       CHECK:   return
//   CHECK-NOT:   list.print
func.func @print(%input: !list.list<i32>) {
  list.print %input : !list.list<i32>
  return
}

// -----

// The elements are printed with the type they are stored with.

// CHECK-LABEL: @print_wide_elements
//       CHECK:   scf.while (%{{.*}} = %{{.*}}) : (!list.list<i64>) -> !list.list<i64>
//       CHECK:   } do {
//       CHECK:     %[[ELEM:.*]] = list.peek_front %{{.*}} : !list.list<i64> -> i64
//       CHECK:     vector.print %[[ELEM]] : i64
func.func @print_wide_elements(%input: !list.list<i64>) {
  list.print %input : !list.list<i64>
  return
}

// -----

// Every `list.print` becomes a loop of its own.

// CHECK-LABEL: @print_twice
// CHECK-SAME:      %[[FIRST:[^:]*]]: !list.list<i32>, %[[SECOND:[^:]*]]: !list.list<i1>
//       CHECK:   scf.while (%{{.*}} = %[[FIRST]])
//       CHECK:     vector.print %{{.*}} : i32
//       CHECK:   }
//       CHECK:   scf.while (%{{.*}} = %[[SECOND]])
//       CHECK:     vector.print %{{.*}} : i1
//       CHECK:   }
//       CHECK:   return
func.func @print_twice(%first: !list.list<i32>, %second: !list.list<i1>) {
  list.print %first : !list.list<i32>
  list.print %second : !list.list<i1>
  return
}
