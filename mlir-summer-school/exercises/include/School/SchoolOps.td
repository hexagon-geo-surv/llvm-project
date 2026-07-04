//===- SchoolOps.td - School dialect ops -------------------*- tablegen -*-===//
//
// Operation definitions of the `school` dialect.
//
// Starter state: the ops carry no traits and no fold/canonicalization hooks.
// That is deliberate -- Exercise 3 adds them one by one and observes what each
// hook buys us.
//
//===----------------------------------------------------------------------===//

#ifndef SCHOOL_OPS
#define SCHOOL_OPS

include "School/SchoolDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// school.max
//===----------------------------------------------------------------------===//

// TODO(exercise 3.1): Edit the trait list. See exercise3.md, checkpoint 1.
// TODO(exercise 3.4, stretch): Add one more trait. See exercise3.md.
def School_MaxOp : School_Op<"max", []> {
  let summary = "signed maximum of two 32-bit integers";
  let description = [{
    Returns the larger of its two operands, comparing them as signed 32-bit
    integers.

    Example:

    ```mlir
    %m = school.max %a, %b : i32
    ```
  }];

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  // TODO(exercise 3.2): Declare the fold hook here, then implement it in
  // lib/School/SchoolOps.cpp. See exercise3.md, checkpoint 2.

  // TODO(exercise 3.5, stretch): Declare the canonicalization-patterns hook
  // here, then implement it in lib/School/SchoolOps.cpp. See exercise3.md.
}

//===----------------------------------------------------------------------===//
// school.mac
//===----------------------------------------------------------------------===//

// TODO(exercise 3.1): Edit the trait list. See exercise3.md, checkpoint 1.
def School_MacOp : School_Op<"mac", []> {
  let summary = "32-bit integer multiply-accumulate: lhs * rhs + acc";
  let description = [{
    Multiplies the first two operands and adds the third:
    `result = lhs * rhs + acc` (all 32-bit integers, wrapping on overflow).

    Example:

    ```mlir
    %r = school.mac %a, %b, %c : i32
    ```
  }];

  let arguments = (ins I32:$lhs, I32:$rhs, I32:$acc);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($result)";
}

#endif // SCHOOL_OPS
