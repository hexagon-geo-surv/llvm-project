// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// Test dense elements attribute with custom element type using DenseElementTypeInterface.
// Uses the new type-first syntax: dense<TYPE : [ATTR, ...]>
// Note: The type is embedded in the attribute, so it's not printed again at the end.

// CHECK-LABEL: func @dense_custom_element_type
func.func @dense_custom_element_type() {
  // CHECK: "test.dummy"() {attr = dense<tensor<3x!test.dense_element> : [1 : i32, 2 : i32, 3 : i32]>}
  "test.dummy"() {attr = dense<tensor<3x!test.dense_element> : [1 : i32, 2 : i32, 3 : i32]>} : () -> ()
  return
}

// CHECK-LABEL: func @dense_custom_element_type_2d
func.func @dense_custom_element_type_2d() {
  // CHECK: "test.dummy"() {attr = dense<tensor<2x2x!test.dense_element> : {{\[}}{{\[}}1 : i32, 2 : i32], [3 : i32, 4 : i32]]>}
  "test.dummy"() {attr = dense<tensor<2x2x!test.dense_element> : [[1 : i32, 2 : i32], [3 : i32, 4 : i32]]>} : () -> ()
  return
}

// CHECK-LABEL: func @dense_custom_element_splat
func.func @dense_custom_element_splat() {
  // CHECK: "test.dummy"() {attr = dense<tensor<4x!test.dense_element> : 42 : i32>}
  "test.dummy"() {attr = dense<tensor<4x!test.dense_element> : 42 : i32>} : () -> ()
  return
}

// CHECK-LABEL func @dense_i32_1d
func.func @dense_i32_1d() {
  // The default assembly format for int, index, float, complex element types is
  // the literal-first syntax. Such a dense elements attribute can be parsed
  // with the type-first syntax, but it will come back with the literal-first
  // syntax.
  // CHECK: "test.dummy"() {attr = dense<[1, 2, 3]> : tensor<3xi32>} : () -> ()
  "test.dummy"() {attr = dense<tensor<3xi32> : [1 : i32, 2 : i32, 3 : i32]>} : () -> ()
  return
}
