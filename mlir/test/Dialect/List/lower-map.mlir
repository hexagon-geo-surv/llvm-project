// RUN: mlir-opt %s -split-input-file -list-lower-map | FileCheck %s

// CHECK-LABEL: @map
// CHECK-SAME:      %[[INPUT:.*]]: !list.list<i32>
//       CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:   %[[EMPTY:.*]] = list.from_elements : () -> !list.list<i64>
//       CHECK:   %[[LOOP:.*]]:2 = scf.while (%[[ARG:.*]] = %[[INPUT]], %[[ACC:.*]] = %[[EMPTY]]) : (!list.list<i32>, !list.list<i64>) -> (!list.list<i32>, !list.list<i64>) {
//       CHECK:     %[[LEN:.*]] = list.length %[[ARG]] : !list.list<i32> -> i32
//       CHECK:     %[[COND:.*]] = arith.cmpi ne, %[[LEN]], %[[C0]] : i32
//       CHECK:     scf.condition(%[[COND]]) %[[ARG]], %[[ACC]] : !list.list<i32>, !list.list<i64>
//       CHECK:   } do {
//       CHECK:   ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i64>):
//       CHECK:     %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
//       CHECK:     %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
//       CHECK:     %[[EXT:.*]] = arith.extsi %[[ELEM]] : i32 to i64
//       CHECK:     %[[LONGER:.*]] = list.push_back %[[ACC]], %[[EXT]] : !list.list<i64>
//       CHECK:     scf.yield %[[REST]], %[[LONGER]] : !list.list<i32>, !list.list<i64>
//       CHECK:   }
//       CHECK:   return %[[LOOP]]#1 : !list.list<i64>
func.func @map(%input: !list.list<i32>) -> !list.list<i64> {
  %mapped = list.map %input with (%element : i32) -> i64 {
    %ext = arith.extsi %element : i32 to i64
    list.yield %ext : i64
  }
  return %mapped : !list.list<i64>
}

// -----

// The body of the map may yield the mapped element itself.

// CHECK-LABEL: @identity
//       CHECK:   scf.while
//       CHECK:   } do {
//       CHECK:   ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i32>):
//       CHECK:     %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
//       CHECK:     %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
//       CHECK:     %[[LONGER:.*]] = list.push_back %[[ACC]], %[[ELEM]] : !list.list<i32>
//       CHECK:     scf.yield %[[REST]], %[[LONGER]] : !list.list<i32>, !list.list<i32>
func.func @identity(%input: !list.list<i32>) -> !list.list<i32> {
  %mapped = list.map %input with (%element : i32) -> i32 {
    list.yield %element : i32
  }
  return %mapped : !list.list<i32>
}

// -----

// The body of the map may use values defined outside of it.

// CHECK-LABEL: @outer_value
// CHECK-SAME:      %[[INPUT:[^:]*]]: !list.list<i32>, %[[FACTOR:[^:]*]]: i32
//       CHECK:   scf.while (%{{.*}} = %[[INPUT]],
//       CHECK:   } do {
//       CHECK:   ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i32>):
//       CHECK:     %[[ELEM:.*]] = list.peek_front %[[ARG]]
//       CHECK:     %[[REST:.*]] = list.pop_front %[[ARG]]
//       CHECK:     %[[SCALED:.*]] = arith.muli %[[ELEM]], %[[FACTOR]] : i32
//       CHECK:     %[[LONGER:.*]] = list.push_back %[[ACC]], %[[SCALED]]
//       CHECK:     scf.yield %[[REST]], %[[LONGER]]
func.func @outer_value(%input: !list.list<i32>, %factor: i32)
    -> !list.list<i32> {
  %mapped = list.map %input with (%element : i32) -> i32 {
    %scaled = arith.muli %element, %factor : i32
    list.yield %scaled : i32
  }
  return %mapped : !list.list<i32>
}

// -----

// Two maps become two loops, the second one iterating over the result of the
// first one.

// CHECK-LABEL: @consecutive_maps
// CHECK-SAME:      %[[INPUT:.*]]: !list.list<i32>
//       CHECK:   %[[FIRST:.*]]:2 = scf.while (%{{.*}} = %[[INPUT]],
//       CHECK:   } do {
//       CHECK:     list.peek_front
//       CHECK:   }
//       CHECK:   %[[SECOND:.*]]:2 = scf.while (%{{.*}} = %[[FIRST]]#1,
//       CHECK:   } do {
//       CHECK:     list.peek_front
//       CHECK:   }
//       CHECK:   return %[[SECOND]]#1
func.func @consecutive_maps(%input: !list.list<i32>) -> !list.list<i32> {
  %first = list.map %input with (%a : i32) -> i32 {
    list.yield %a : i32
  }
  %second = list.map %first with (%b : i32) -> i32 {
    list.yield %b : i32
  }
  return %second : !list.list<i32>
}

// -----

// A map nested in the body of another map becomes a loop nested in a loop.

// CHECK-LABEL: @nested_map
//       CHECK:   scf.while
//       CHECK:   } do {
//       CHECK:     %[[ELEM:.*]] = list.peek_front
//       CHECK:     %[[REST:.*]] = list.pop_front
//       CHECK:     %[[RANGE:.*]] = list.range %[[ELEM]] to %[[ELEM]]
//       CHECK:     %[[INNER:.*]]:2 = scf.while (%{{.*}} = %[[RANGE]],
//       CHECK:     } do {
//       CHECK:       list.peek_front
//       CHECK:     }
//       CHECK:     %[[LEN:.*]] = list.length %[[INNER]]#1
//       CHECK:     %[[LONGER:.*]] = list.push_back %{{.*}}, %[[LEN]]
//       CHECK:     scf.yield %[[REST]], %[[LONGER]]
func.func @nested_map(%input: !list.list<i32>) -> !list.list<i32> {
  %mapped = list.map %input with (%element : i32) -> i32 {
    %range = list.range %element to %element : !list.list<i32>
    %inner = list.map %range with (%e : i32) -> i32 {
      list.yield %e : i32
    }
    %len = list.length %inner : !list.list<i32> -> i32
    list.yield %len : i32
  }
  return %mapped : !list.list<i32>
}
