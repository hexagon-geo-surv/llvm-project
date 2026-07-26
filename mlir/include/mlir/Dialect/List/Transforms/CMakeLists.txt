set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name List)
add_mlir_dialect_tablegen_target(MLIRListTransformsIncGen)

add_mlir_doc(Passes ListPasses ./ -gen-pass-doc)
