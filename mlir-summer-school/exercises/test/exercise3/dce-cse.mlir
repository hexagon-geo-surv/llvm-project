// RUN: school-opt %s -canonicalize | FileCheck %s --check-prefix=DCE
// RUN: school-opt %s -cse | FileCheck %s --check-prefix=CSE

// Exercise 3 checkpoint 1: DCE and CSE are gated on side-effect information.
// In the starter state, school.max has no effect annotation, so MLIR must
// conservatively assume it does something observable: -canonicalize keeps the
// dead op and -cse refuses to merge the duplicates. Adding `Pure` in ODS
// turns both tests green -- without writing a single line of pass code.

// A school.max whose result is never used. Only a side-effect-free op may be
// deleted.
// DCE-LABEL: func.func @dead_max
// DCE-SAME:  (%[[X:.+]]: i32, %[[Y:.+]]: i32)
func.func @dead_max(%x: i32, %y: i32) -> i32 {
  // DCE-NOT: school.max
  %dead = school.max %x, %y : i32
  // DCE: return %[[X]]
  return %x : i32
}

// Two identical school.max ops. Only side-effect-free ops are eligible
// for CSE.
// CSE-LABEL: func.func @duplicate_max
// CSE-SAME:  (%[[X:.+]]: i32, %[[Y:.+]]: i32)
func.func @duplicate_max(%x: i32, %y: i32) -> (i32, i32) {
  // CSE: %[[M:.+]] = school.max %[[X]], %[[Y]] : i32
  %a = school.max %x, %y : i32
  // CSE-NOT: school.max
  %b = school.max %x, %y : i32
  // CSE: return %[[M]], %[[M]]
  return %a, %b : i32, i32
}
