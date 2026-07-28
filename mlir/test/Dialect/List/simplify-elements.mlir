// RUN: mlir-opt %s -split-input-file -list-simplify | FileCheck %s

// Every element is extracted with a `list.peek_front`, from the front of the
// list to its back.

// CHECK-LABEL: @unroll_get_elements(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[FIRST:.*]] = list.peek_front %[[LI]] : !list.list<i32> -> i32
// CHECK-NEXT:    %[[REST1:.*]] = list.pop_front %[[LI]] : !list.list<i32>
// CHECK-NEXT:    %[[SECOND:.*]] = list.peek_front %[[REST1]] : !list.list<i32> -> i32
// CHECK-NEXT:    %[[REST2:.*]] = list.pop_front %[[REST1]] : !list.list<i32>
// CHECK-NEXT:    %[[THIRD:.*]] = list.peek_front %[[REST2]] : !list.list<i32> -> i32
// The last `list.pop_front` has no users: this pass does not remove dead code.
// CHECK-NEXT:    list.pop_front %[[REST2]] : !list.list<i32>
// CHECK-NEXT:    return %[[FIRST]], %[[SECOND]], %[[THIRD]] : i32, i32, i32
// CHECK-NOT:     list.get_elements
func.func @unroll_get_elements(%li: !list.list<i32>) -> (i32, i32, i32) {
  %a, %b, %c = list.get_elements %li : (!list.list<i32>) -> (i32, i32, i32)
  return %a, %b, %c : i32, i32, i32
}

// -----

// A `list.get_elements` without results extracts nothing and is removed, which
// is what terminates the unrolling.

// CHECK-LABEL: @get_no_elements(
// CHECK-NEXT:    return
// CHECK-NOT:     list.get_elements
func.func @get_no_elements(%li: !list.list<i32>) {
  list.get_elements %li : (!list.list<i32>) -> ()
  return
}

// -----

// Every element is appended with a `list.push_back`, from the front of the list
// to its back.

// CHECK-LABEL: @unroll_from_elements(
// CHECK-SAME:      %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32
// CHECK:         %[[EMPTY:.*]] = list.empty : !list.list<i32>
// CHECK-NEXT:    %[[ONE:.*]] = list.push_back %[[EMPTY]], %[[A]] : !list.list<i32>
// CHECK-NEXT:    %[[TWO:.*]] = list.push_back %[[ONE]], %[[B]] : !list.list<i32>
// CHECK-NEXT:    %[[THREE:.*]] = list.push_back %[[TWO]], %[[C]] : !list.list<i32>
// CHECK-NEXT:    return %[[THREE]] : !list.list<i32>
// CHECK-NOT:     list.from_elements
func.func @unroll_from_elements(%a: i32, %b: i32, %c: i32) -> !list.list<i32> {
  %li = list.from_elements %a, %b, %c : (i32, i32, i32) -> !list.list<i32>
  return %li : !list.list<i32>
}

// -----

// A `list.from_elements` without any element becomes a `list.empty`, which is
// what terminates the unrolling.

// CHECK-LABEL: @from_no_elements(
// CHECK:         %[[EMPTY:.*]] = list.empty : !list.list<i64>
// CHECK-NEXT:    return %[[EMPTY]] : !list.list<i64>
// CHECK-NOT:     list.from_elements
func.func @from_no_elements() -> !list.list<i64> {
  %li = list.from_elements : () -> !list.list<i64>
  return %li : !list.list<i64>
}

// -----

// Nothing is known about the operand list, so the push stays as it is.

// CHECK-LABEL: @push_back_on_opaque_list(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>, %[[ITEM:[^:]*]]: i32
// CHECK:         %[[LONGER:.*]] = list.push_back %[[LI]], %[[ITEM]] : !list.list<i32>
// CHECK:         return %[[LONGER]] : !list.list<i32>
func.func @push_back_on_opaque_list(%li: !list.list<i32>, %item: i32)
    -> !list.list<i32> {
  %longer = list.push_back %li, %item : !list.list<i32>
  return %longer : !list.list<i32>
}

// -----

// A list that is built up and taken apart again is expressed entirely in terms
// of `list.empty`, `list.push_back`, `list.peek_front` and `list.pop_front`.

// CHECK-LABEL: @get_elements_of_from_elements(
// CHECK-SAME:      %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK:         %[[EMPTY:.*]] = list.empty : !list.list<i32>
// CHECK-NEXT:    %[[ONE:.*]] = list.push_back %[[EMPTY]], %[[A]] : !list.list<i32>
// CHECK-NEXT:    %[[TWO:.*]] = list.push_back %[[ONE]], %[[B]] : !list.list<i32>
// CHECK-NEXT:    %[[FIRST:.*]] = list.peek_front %[[TWO]] : !list.list<i32> -> i32
// CHECK-NEXT:    %[[REST:.*]] = list.pop_front %[[TWO]] : !list.list<i32>
// CHECK-NEXT:    %[[SECOND:.*]] = list.peek_front %[[REST]] : !list.list<i32> -> i32
// CHECK:         return %[[FIRST]], %[[SECOND]] : i32, i32
func.func @get_elements_of_from_elements(%a: i32, %b: i32) -> (i32, i32) {
  %li = list.from_elements %a, %b : (i32, i32) -> !list.list<i32>
  %0, %1 = list.get_elements %li : (!list.list<i32>) -> (i32, i32)
  return %0, %1 : i32, i32
}
