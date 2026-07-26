// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// Two consecutive maps become a single map that applies both computations.

// CHECK-LABEL: @merge_maps(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[MERGED:.*]] = list.map %[[LI]] with (%[[ELEM:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[DOUBLED:.*]] = arith.addi %[[ELEM]], %[[ELEM]] : i32
// CHECK-NEXT:      %[[EXTENDED:.*]] = arith.extsi %[[DOUBLED]] : i32 to i64
// CHECK-NEXT:      list.yield %[[EXTENDED]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[MERGED]] : !list.list<i64>
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

// Merging is applied repeatedly: a chain of three maps becomes one.

// CHECK-LABEL: @merge_three_maps(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[MERGED:.*]] = list.map %[[LI]] with (%[[ELEM:.*]]: i32) -> i32 {
// CHECK-NEXT:      %[[SQUARED:.*]] = arith.muli %[[ELEM]], %[[ELEM]] : i32
// CHECK-NEXT:      list.yield %[[SQUARED]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[MERGED]] : !list.list<i32>
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
// that map has more than one user.

// CHECK-LABEL: @producer_has_other_users(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[DOUBLED:.*]] = list.map %[[LI]] with (%{{.*}}: i32) -> i32 {
// CHECK:         %[[COPY:.*]] = list.map %[[DOUBLED]] with (%{{.*}}: i32) -> i32 {
// CHECK:         return %[[DOUBLED]], %[[COPY]]
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

// The length of a mapped list is the length of the list it is mapped from. The
// map itself stays behind: it may have side effects or other users, and this
// pass does not remove dead code.

// CHECK-LABEL: @length_of_map(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[LEN:.*]] = list.length %[[LI]] : !list.list<i32> -> i32
// CHECK-NEXT:    return %[[LEN]] : i32
func.func @length_of_map(%li: !list.list<i32>) -> i32 {
  %extended = list.map %li with (%a : i32) -> i64 {
    %0 = arith.extsi %a : i32 to i64
    list.yield %0 : i64
  }
  %len = list.length %extended : !list.list<i64> -> i32
  return %len : i32
}

// -----

// Both patterns interact: the maps are merged and the length then reads through
// the merged map.

// CHECK-LABEL: @length_of_nested_maps(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[LEN:.*]] = list.length %[[LI]] : !list.list<i32> -> i32
// CHECK-NEXT:    return %[[LEN]] : i32
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

// Merging preserves the order of side effects of both bodies.

// CHECK-LABEL: @merge_keeps_side_effects(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>, %[[OTHER:[^:]*]]: !list.list<i32>
// CHECK:         %[[MERGED:.*]] = list.map %[[LI]] with (%[[ELEM:.*]]: i32) -> i32 {
// CHECK-NEXT:      list.print %[[OTHER]] : !list.list<i32>
// CHECK-NEXT:      %[[DOUBLED:.*]] = arith.addi %[[ELEM]], %[[ELEM]] : i32
// CHECK-NEXT:      list.yield %[[DOUBLED]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[MERGED]] : !list.list<i32>
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
