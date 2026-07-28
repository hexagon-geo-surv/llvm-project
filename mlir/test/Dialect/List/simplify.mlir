// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// The simplifications are applied before the lowerings, so the two maps are
// merged first and a single loop applies both computations to every element.

// CHECK-LABEL: @merge_maps(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[LOOP:.*]]:2 = scf.while (%{{.*}} = %[[LI]],
// CHECK:         } do {
// CHECK-NEXT:    ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i64>):
// CHECK-NEXT:      %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
// CHECK-NEXT:      %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
// CHECK-NEXT:      %[[DOUBLED:.*]] = arith.addi %[[ELEM]], %[[ELEM]] : i32
// CHECK-NEXT:      %[[EXTENDED:.*]] = arith.extsi %[[DOUBLED]] : i32 to i64
// CHECK-NEXT:      %[[LONGER:.*]] = list.push_back %[[ACC]], %[[EXTENDED]] : !list.list<i64>
// CHECK-NEXT:      scf.yield %[[REST]], %[[LONGER]] : !list.list<i32>, !list.list<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[LOOP]]#1 : !list.list<i64>
func.func @merge_maps(%li: !list.list<i32>) -> !list.list<i64> {
  %doubled = list.map %li with (%a : i32) -> i32 {
    %0 = arith.addi %a, %a : i32
    list.yield %0 : i32
  }
  %extended = list.map %doubled with (%b : i32) -> i64 {
    %1 = arith.extsi %b : i32 to i64
    list.yield %1 : i64
  }
  return %extended : !list.list<i64>
}

// -----

// Merging is applied repeatedly: a chain of three maps becomes one loop. The
// two identity maps leave nothing behind in its body.

// CHECK-LABEL: @merge_three_maps(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[LOOP:.*]]:2 = scf.while (%{{.*}} = %[[LI]],
// CHECK:         } do {
// CHECK-NEXT:    ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i32>):
// CHECK-NEXT:      %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
// CHECK-NEXT:      %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
// CHECK-NEXT:      %[[SQUARED:.*]] = arith.muli %[[ELEM]], %[[ELEM]] : i32
// CHECK-NEXT:      %[[LONGER:.*]] = list.push_back %[[ACC]], %[[SQUARED]] : !list.list<i32>
// CHECK-NEXT:      scf.yield %[[REST]], %[[LONGER]] : !list.list<i32>, !list.list<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[LOOP]]#1 : !list.list<i32>
func.func @merge_three_maps(%li: !list.list<i32>) -> !list.list<i32> {
  %identity = list.map %li with (%a : i32) -> i32 {
    list.yield %a : i32
  }
  %squared = list.map %identity with (%b : i32) -> i32 {
    %0 = arith.muli %b, %b : i32
    list.yield %0 : i32
  }
  %again = list.map %squared with (%c : i32) -> i32 {
    list.yield %c : i32
  }
  return %again : !list.list<i32>
}

// -----

// Merging would duplicate the body of the first map, so it is not applied when
// that map has more than one user. Both maps become a loop of their own, the
// second one iterating over the result of the first one.

// CHECK-LABEL: @producer_has_other_users(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[FIRST:.*]]:2 = scf.while (%{{.*}} = %[[LI]],
// CHECK:         } do {
// CHECK:           arith.addi
// CHECK:         }
// CHECK:         %[[SECOND:.*]]:2 = scf.while (%{{.*}} = %[[FIRST]]#1,
// CHECK:         } do {
// CHECK:           list.peek_front
// CHECK:         }
// CHECK:         return %[[FIRST]]#1, %[[SECOND]]#1
func.func @producer_has_other_users(%li: !list.list<i32>)
    -> (!list.list<i32>, !list.list<i32>) {
  %doubled = list.map %li with (%a : i32) -> i32 {
    %0 = arith.addi %a, %a : i32
    list.yield %0 : i32
  }
  %copy = list.map %doubled with (%b : i32) -> i32 {
    list.yield %b : i32
  }
  return %doubled, %copy : !list.list<i32>, !list.list<i32>
}

// -----

// The length of a mapped list is the length of the list it is mapped from, so
// the loop that counts the elements iterates over the operand of the map. The
// map itself stays behind: its body may have side effects, and this pass does
// not remove dead code.

// CHECK-LABEL: @length_of_map(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// The loop of the map, whose result is not used anymore.
// CHECK:         scf.while (%{{.*}} = %[[LI]], %{{.*}}) : (!list.list<i32>, !list.list<i64>)
// CHECK:         }
// CHECK:         %[[COUNT:.*]]:2 = scf.while (%{{.*}} = %[[LI]], %{{.*}} = %[[C0]]) : (!list.list<i32>, i32)
// CHECK:         } do {
// CHECK:           list.pop_front
// CHECK:           arith.addi
// CHECK:         }
// CHECK:         return %[[COUNT]]#1 : i32
func.func @length_of_map(%li: !list.list<i32>) -> i32 {
  %extended = list.map %li with (%a : i32) -> i64 {
    %0 = arith.extsi %a : i32 to i64
    list.yield %0 : i64
  }
  %len = list.length %extended : !list.list<i64> -> i32
  return %len : i32
}

// -----

// Both simplifications interact: the maps are merged and the length then reads
// through the merged map, so the elements of the operand list are counted.

// CHECK-LABEL: @length_of_nested_maps(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// A single loop is left for the two merged maps.
// CHECK:         scf.while (%{{.*}} = %[[LI]],
// CHECK:           arith.muli
// CHECK:         }
// CHECK:         %[[COUNT:.*]]:2 = scf.while (%{{.*}} = %[[LI]], %{{.*}} = %[[C0]]) : (!list.list<i32>, i32)
// CHECK:         return %[[COUNT]]#1 : i32
func.func @length_of_nested_maps(%li: !list.list<i32>) -> i32 {
  %squared = list.map %li with (%a : i32) -> i32 {
    %0 = arith.muli %a, %a : i32
    list.yield %0 : i32
  }
  %copy = list.map %squared with (%b : i32) -> i32 {
    list.yield %b : i32
  }
  %len = list.length %copy : !list.list<i32> -> i32
  return %len : i32
}

// -----

// Merging preserves the order of side effects of both bodies: the list of the
// print is walked before the element of the outer list is doubled.

// CHECK-LABEL: @merge_keeps_side_effects(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>, %[[OTHER:[^:]*]]: !list.list<i32>
// CHECK:         %[[LOOP:.*]]:2 = scf.while (%{{.*}} = %[[LI]],
// CHECK:         } do {
// CHECK-NEXT:    ^bb0(%[[ARG:.*]]: !list.list<i32>, %[[ACC:.*]]: !list.list<i32>):
// CHECK-NEXT:      %[[ELEM:.*]] = list.peek_front %[[ARG]] : !list.list<i32> -> i32
// CHECK-NEXT:      %[[REST:.*]] = list.pop_front %[[ARG]] : !list.list<i32>
// CHECK-NEXT:      scf.while (%{{.*}} = %[[OTHER]])
// CHECK:             vector.print
// CHECK:           }
// CHECK-NEXT:      %[[DOUBLED:.*]] = arith.addi %[[ELEM]], %[[ELEM]] : i32
// CHECK-NEXT:      %[[LONGER:.*]] = list.push_back %[[ACC]], %[[DOUBLED]] : !list.list<i32>
// CHECK-NEXT:      scf.yield %[[REST]], %[[LONGER]] : !list.list<i32>, !list.list<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[LOOP]]#1 : !list.list<i32>
func.func @merge_keeps_side_effects(%li: !list.list<i32>,
                                    %other: !list.list<i32>)
    -> !list.list<i32> {
  %printed = list.map %li with (%a : i32) -> i32 {
    list.print %other : !list.list<i32>
    list.yield %a : i32
  }
  %doubled = list.map %printed with (%b : i32) -> i32 {
    %0 = arith.addi %b, %b : i32
    list.yield %0 : i32
  }
  return %doubled : !list.list<i32>
}
