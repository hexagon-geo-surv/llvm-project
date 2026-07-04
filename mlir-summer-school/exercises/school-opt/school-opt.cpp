//===- school-opt.cpp - The summer school optimizer driver ------*- C++ -*-===//
//
// An mlir-opt-style tool for the exercises. All the heavy lifting (parsing,
// pass-pipeline handling, printing, all the --mlir-* debugging flags) lives
// in MlirOptMain; this file only decides what is registered.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "School/SchoolDialect.h"
#include "School/SchoolPasses.h"

int main(int argc, char **argv) {
  // Passes available on the command line: the upstream transforms
  // (canonicalize, cse, ...) plus the three exercise passes.
  mlir::registerTransformsPasses();
  mlir::school::registerSchoolPasses();

  // Dialects the tool can *parse*. A small hand-picked set keeps the binary
  // small and the link fast; registerAllDialects() would work too.
  mlir::DialectRegistry registry;
  registry.insert<mlir::school::SchoolDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR summer school exercise driver\n", registry));
}
