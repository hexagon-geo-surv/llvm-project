# school-opt: an mlir-opt-style driver with the school dialect and the
# exercise passes registered. Deliberately does NOT link
# MLIRRegisterAllDialects/MLIRRegisterAllPasses to keep link times short.
set(LIBS
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRSCFDialect
  MLIRSchool
  MLIRTransforms
)
add_llvm_executable(school-opt school-opt.cpp)

llvm_update_compile_flags(school-opt)
target_link_libraries(school-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(school-opt)
