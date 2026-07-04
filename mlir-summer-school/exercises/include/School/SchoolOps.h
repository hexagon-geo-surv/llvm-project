//===- SchoolOps.h - School dialect ops -------------------------*- C++ -*-===//
//
// Pulls in the TableGen-generated op classes (school::MaxOp, school::MacOp).
//
//===----------------------------------------------------------------------===//

#ifndef SCHOOL_SCHOOLOPS_H
#define SCHOOL_SCHOOLOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "School/SchoolOps.h.inc"

#endif // SCHOOL_SCHOOLOPS_H
