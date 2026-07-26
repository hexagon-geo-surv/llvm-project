// RUN: mlir-opt %s -split-input-file -list-simplify-elements | FileCheck %s

// Every element is extracted with a `list.peek_back`, from the back of the list
// to its front.

// CHECK-LABEL: @unroll_get_elements(
// CHECK-SAME:      %[[LI:[^:]*]]: !list.list<i32>
// CHECK:         %[[THIRD:.*]] = list.peek_back %[[LI]] : !list.list<i32> -> i32
// CHECK-NEXT:    %[[REST1:.*]] = list.pop_back %[[LI]] : !list.list<i32>
// CHECK-NEXT:    %[[SECOND:.*]] = list.peek_back %[[REST1]] : !list.list<i32> -> i32
// CHECK-NEXT:    %[[REST2:.*]] = list.pop_back %[[REST1]] : !list.list<i32>
// CHECK-NEXT:    %[[FIRST:.*]] = list.peek_back %[[REST2]] : !list.list<i32> -> i32
// The last `list.pop_back` has no users: this pass does not remove dead code.
// CHECK-NEXT:    list.pop_back %[[REST2]] : !list.list<i32>
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

// CHECK-LABEL: @fold_push_back(
// CHECK-SAME:      %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32
// CHECK:         %[[LONGER:.*]] = list.from_elements %[[A]], %[[B]], %[[C]] : (i32, i32, i32) -> !list.list<i32>
// CHECK:         return %[[LONGER]] : !list.list<i32>
// CHECK-NOT:     list.push_back
func.func @fold_push_back(%a: i32, %b: i32, %c: i32) -> !list.list<i32> {
  %li = list.from_elements %a, %b : (i32, i32) -> !list.list<i32>
  %longer = list.push_back %li, %c : !list.list<i32>
  return %longer : !list.list<i32>
}

// -----

// Folding is applied repeatedly, one pushed item at a time.

// CHECK-LABEL: @fold_two_push_backs(
// CHECK-SAME:      %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32
// CHECK:         %[[LONGER:.*]] = list.from_elements %[[A]], %[[B]], %[[C]] : (i32, i32, i32) -> !list.list<i32>
// CHECK:         return %[[LONGER]] : !list.list<i32>
// CHECK-NOT:     list.push_back
func.func @fold_two_push_backs(%a: i32, %b: i32, %c: i32) -> !list.list<i32> {
  %li = list.from_elements %a : (i32) -> !list.list<i32>
  %longer = list.push_back %li, %b : !list.list<i32>
  %even_longer = list.push_back %longer, %c : !list.list<i32>
  return %even_longer : !list.list<i32>
}

// -----

// Nothing is known about the operand list, so the push stays.

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

// The `list.from_elements` is kept because the unrolled reads use it.

// CHECK-LABEL: @get_elements_of_from_elements(
// CHECK-SAME:      %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK:         %[[LI:.*]] = list.from_elements %[[A]], %[[B]] : (i32, i32) -> !list.list<i32>
// CHECK-NEXT:    %[[SECOND:.*]] = list.peek_back %[[LI]] : !list.list<i32> -> i32
// CHECK-NEXT:    %[[REST:.*]] = list.pop_back %[[LI]] : !list.list<i32>
// CHECK-NEXT:    %[[FIRST:.*]] = list.peek_back %[[REST]] : !list.list<i32> -> i32
// CHECK:         return %[[FIRST]], %[[SECOND]] : i32, i32
func.func @get_elements_of_from_elements(%a: i32, %b: i32) -> (i32, i32) {
  %li = list.from_elements %a, %b : (i32, i32) -> !list.list<i32>
  %0, %1 = list.get_elements %li : (!list.list<i32>) -> (i32, i32)
  return %0, %1 : i32, i32
}
