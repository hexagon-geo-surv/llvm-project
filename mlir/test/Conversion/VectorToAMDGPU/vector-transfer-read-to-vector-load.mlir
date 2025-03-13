// RUN: mlir-opt %s -convert-vector-to-amdgpu --split-input-file | FileCheck %s

func.func @transfer_to_maskedload(%mem : memref<8x8xf32>, %idx : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: func @transfer_to_maskedload
// func.func @transfer_to_maskedload(%arg0: memref<8x8xf32>, %arg1: index, %arg2: vector<4xi1>) -> vector<4xf32> {
//    %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
//    %0 = vector.load %arg0[%arg1, %arg1] : memref<8x8xf32>, vector<4xf32>
//    %1 = arith.select %arg2, %0, %cst : vector<4xi1>, vector<4xf32>
//    return %1 : vector<4xf32>
//  }

