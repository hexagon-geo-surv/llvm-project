add_mlir_dialect_library(MLIRListDialect
  ListDialect.cpp
  ListOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/List

  DEPENDS
  MLIRListOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRSupport
  )
