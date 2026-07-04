//===- SchoolPasses.h - School exercise passes ------------------*- C++ -*-===//
//
// Declarations and registration for the exercise passes. The actual
// declarations are generated from SchoolPasses.td:
//   - GEN_PASS_DECL:         create<PassName>() factory functions
//   - GEN_PASS_REGISTRATION: registerSchoolPasses() (called by school-opt)
//
//===----------------------------------------------------------------------===//

#ifndef SCHOOL_SCHOOLPASSES_H
#define SCHOOL_SCHOOLPASSES_H

#include "School/SchoolDialect.h"
#include "School/SchoolOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::school {
#define GEN_PASS_DECL
#include "School/SchoolPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "School/SchoolPasses.h.inc"
} // namespace mlir::school

#endif // SCHOOL_SCHOOLPASSES_H
