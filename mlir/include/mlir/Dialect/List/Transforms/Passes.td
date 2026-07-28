//===-- Passes.td - List transformation definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LIST_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_LIST_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ListSimplify : Pass<"list-simplify"> {
  let summary = "Simplify list operations and lower them to loops";
  let description = [{
    Applies all patterns of the `list` dialect at once. What is left of a
    program afterwards are loops over the primitives of the dialect:
    `list.empty`, `list.is_empty`, `list.peek_front`, `list.pop_front`,
    `list.push_back` and `list.push_front`.

    Two patterns simplify operations without lowering them:

    - Two consecutive `list.map` operations are merged into a single one that
      applies both computations to every element:
      `list.map(list.map(%x))` becomes `list.map(%x)`.
    - A `list.length` reads through a `list.map`, because mapping does not
      change the number of elements of a list:
      `list.length(list.map(%x))` becomes `list.length(%x)`.

    Three patterns express the operations that convert between lists and their
    elements in terms of the primitives:

    - A `list.get_elements` operation is replaced by a `list.peek_front` and a
      `list.pop_front` for its first result, and a `list.get_elements` for the
      remaining results. Applied repeatedly, this turns every
      `list.get_elements` into a sequence of `list.peek_front` and
      `list.pop_front` operations.
    - A `list.from_elements` operation is replaced by a `list.from_elements` for
      all but its last element and a `list.push_back` for its last one. Applied
      repeatedly, this turns every `list.from_elements` into a sequence of
      `list.push_back` operations on a `list.empty`.
    - A `list.from_elements` without any element becomes a `list.empty`.

    The remaining patterns replace the operations that walk over a whole list
    with a loop that processes one element per iteration:

    - A `list.map` becomes an `scf.while` loop that carries the elements that
      still have to be mapped and the mapped elements collected so far. Every
      iteration takes the first element that is left, applies the body of the
      `list.map` to it and appends the mapped element to the back of the result
      list.
    - A `list.range` becomes an `scf.for` loop that iterates over the integers
      of the range and appends them to a list that starts out empty. The loop
      runs zero times for an empty range, so no special case is needed for it.
    - A `list.reverse` becomes an `scf.while` loop that moves the elements one
      at a time, taking them off the front of the input list and prepending
      them to the result list, which turns their order around. A `list.reverse`
      of a `list.empty` becomes a `list.empty` instead, because there is
      nothing to move.
    - A `list.length` becomes an `scf.while` loop that takes the elements off
      the list one at a time and counts them, so that the length of a list does
      not have to be a primitive of the dialect.
    - A `list.print` becomes an `scf.while` loop that takes the elements off the
      list one at a time and prints each of them with a `vector.print`.

    The simplifications are tried before the lowerings, so that a `list.map`
    that can be merged into another one is merged before it becomes a loop.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "list::ListDialect",
    "scf::SCFDialect",
    "vector::VectorDialect"
  ];
}

#endif // MLIR_DIALECT_LIST_TRANSFORMS_PASSES_TD
