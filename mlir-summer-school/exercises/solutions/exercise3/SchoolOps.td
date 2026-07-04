//===- SchoolOps.td - School dialect ops -------------------*- tablegen -*-===//
//
// Reference solution for Exercise 3 (all checkpoints and stretch goals):
//   3.1  Pure on both ops           -> DCE and CSE work
//   3.2  hasFolder on school.max    -> max(x,x) -> x
//   3.3  (constant materializer: see SchoolDialect.td / SchoolDialect.cpp)
//   3.4  Commutative on school.max  -> constants move to the rhs for free
//   3.5  hasCanonicalizer           -> reassociation pattern (SchoolOps.cpp)
//
//===----------------------------------------------------------------------===//

#ifndef SCHOOL_OPS
#define SCHOOL_OPS

include "School/SchoolDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// school.max
//===----------------------------------------------------------------------===//

// Pure = NoMemoryEffect + AlwaysSpeculatable: tells DCE the op may be
// deleted when unused, and CSE that duplicates may be merged.
// Commutative additionally enables trait folding that moves constant
// operands to the right -- which is why the folder and the reassociation
// pattern only ever inspect the rhs.
def School_MaxOp : School_Op<"max", [Pure, Commutative]> {
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

  // Declares `OpFoldResult MaxOp::fold(FoldAdaptor)` (single-result form),
  // defined in SchoolOps.cpp.
  let hasFolder = 1;

  // Declares `static void MaxOp::getCanonicalizationPatterns(...)`, defined
  // in SchoolOps.cpp. -canonicalize picks these patterns up automatically.
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// school.mac
//===----------------------------------------------------------------------===//

// Pure only: mac is NOT Commutative -- its third operand plays a different
// role than the first two (a*b+c != c*b+a in general).
def School_MacOp : School_Op<"mac", [Pure]> {
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
