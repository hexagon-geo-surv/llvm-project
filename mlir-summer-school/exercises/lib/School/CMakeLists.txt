add_mlir_dialect_library(MLIRSchool
  SchoolDialect.cpp
  SchoolOps.cpp
  StrengthReduce.cpp
  Peephole.cpp
  ConvertSchoolToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/School

  DEPENDS
  MLIRSchoolOpsIncGen
  MLIRSchoolPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
