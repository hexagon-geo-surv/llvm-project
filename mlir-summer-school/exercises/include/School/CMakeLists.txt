# Generate the C++ classes for the dialect and its ops from the ODS files
# (SchoolOps.td pulls in SchoolDialect.td). Creates target MLIRSchoolOpsIncGen.
add_mlir_dialect(SchoolOps school)

# Generate the pass base classes from SchoolPasses.td.
# `-name School` determines the group registration function name:
# mlir::school::registerSchoolPasses().
set(LLVM_TARGET_DEFINITIONS SchoolPasses.td)
mlir_tablegen(SchoolPasses.h.inc --gen-pass-decls -name School)
add_public_tablegen_target(MLIRSchoolPassesIncGen)
