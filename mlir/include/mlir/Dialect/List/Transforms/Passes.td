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
  let summary = "Simplify list operations";
  let description = [{
    Applies simplification patterns to `list` dialect operations:

    - Two consecutive `list.map` operations are merged into a single one that
      applies both computations to every element:
      `list.map(list.map(%x))` becomes `list.map(%x)`.
    - A `list.length` reads through a `list.map`, because mapping does not
      change the number of elements of a list:
      `list.length(list.map(%x))` becomes `list.length(%x)`.
  }];

  let dependentDialects = ["list::ListDialect"];
}

def ListSimplifyElements : Pass<"list-simplify-elements"> {
  let summary = "Simplify operations that take lists apart and build them up";
  let description = [{
    Applies simplification patterns to `list` dialect operations that convert
    between lists and their elements:

    - A `list.get_elements` operation is replaced by a `list.peek_back` and a
      `list.pop_back` for its last result, and a `list.get_elements` for the
      remaining results. Applied repeatedly, this turns every
      `list.get_elements` into a sequence of `list.peek_back` and
      `list.pop_back` operations.
    - A `list.push_back` whose operand list is built by a `list.from_elements`
      becomes a single `list.from_elements` with the pushed item appended.
  }];

  let dependentDialects = ["list::ListDialect"];
}

def ListLowerMap : Pass<"list-lower-map"> {
  let summary = "Lower list.map operations to scf.while loops";
  let description = [{
    Replaces every `list.map` operation with an `scf.while` loop that carries two
    lists: the elements that still have to be mapped and the mapped elements
    collected so far. Every iteration takes the first element that is left,
    applies the body of the `list.map` to it and appends the mapped element to
    the back of the result list. The loop stops once no element is left, at which
    point the second loop-carried list holds the result of the `list.map`.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "list::ListDialect",
    "scf::SCFDialect"
  ];
}

def ListLowerRange : Pass<"list-lower-range"> {
  let summary = "Lower list.range operations to scf.for loops";
  let description = [{
    Replaces every `list.range` operation with an `scf.for` loop that iterates
    over the integers of the range and appends them to a list that starts out
    empty. The loop runs zero times for an empty range, so no special case is
    needed for it.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "list::ListDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_DIALECT_LIST_TRANSFORMS_PASSES_TD
