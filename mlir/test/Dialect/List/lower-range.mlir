// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// CHECK-LABEL: @range
// CHECK-SAME:      %[[LB:[^:]*]]: i32, %[[UB:[^:]*]]: i32
//       CHECK:   %[[C1:.*]] = arith.constant 1 : i32
//       CHECK:   %[[EMPTY:.*]] = list.empty : !list.list<i32>
//       CHECK:   %[[LOOP:.*]] = scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[C1]] iter_args(%[[COLLECTED:.*]] = %[[EMPTY]]) -> (!list.list<i32>) : i32 {
//       CHECK:     %[[LONGER:.*]] = list.push_back %[[COLLECTED]], %[[I]] : !list.list<i32>
//       CHECK:     scf.yield %[[LONGER]] : !list.list<i32>
//       CHECK:   }
//       CHECK:   return %[[LOOP]] : !list.list<i32>
func.func @range(%lb: i32, %ub: i32) -> !list.list<i32> {
  %li = list.range %lb to %ub : !list.list<i32>
  return %li : !list.list<i32>
}

// -----

// The bounds of a `list.range` are always `i32`, so a list of elements of
// another type cannot be built from the induction variable of the loop.

// CHECK-LABEL: @wider_elements
//       CHECK:   list.range
//   CHECK-NOT:   scf.for
func.func @wider_elements(%lb: i32, %ub: i32) -> !list.list<i64> {
  %li = list.range %lb to %ub : !list.list<i64>
  return %li : !list.list<i64>
}

// -----

// A range in the body of a map is lowered as well, ending up in the loop that
// the map becomes.

// CHECK-LABEL: @range_in_map
// CHECK-SAME:      %[[LB:[^:]*]]: i32, %[[UB:[^:]*]]: i32
//       CHECK:   %[[C1:.*]] = arith.constant 1 : i32
//       CHECK:   %[[OUTER:.*]] = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[C1]]
//       CHECK:     list.push_back
//       CHECK:   }
//       CHECK:   scf.while (%{{.*}} = %[[OUTER]],
//       CHECK:   } do {
//       CHECK:     %[[ELEM:.*]] = list.peek_front
//       CHECK:     list.pop_front
//       CHECK:     %[[INNER:.*]] = scf.for %{{.*}} = %[[ELEM]] to %[[ELEM]] step %[[C1]]
//       CHECK:       list.push_back
//       CHECK:     }
// The length of the range is counted by a loop of its own.
//       CHECK:     %[[LEN:.*]]:2 = scf.while (%{{.*}} = %[[INNER]], %{{.*}}) : (!list.list<i32>, i32)
//       CHECK:     list.push_back %{{.*}}, %[[LEN]]#1
func.func @range_in_map(%lb: i32, %ub: i32) -> !list.list<i32> {
  %li = list.range %lb to %ub : !list.list<i32>
  %mapped = list.map %li with (%element : i32) -> i32 {
    %inner = list.range %element to %element : !list.list<i32>
    %len = list.length %inner : !list.list<i32> -> i32
    list.yield %len : i32
  }
  return %mapped : !list.list<i32>
}
