add_mlir_dialect_library(MLIRListTransforms
  ListSimplify.cpp
  ListSimplifyElements.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/List/Transforms

  DEPENDS
  MLIRListTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRListDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
