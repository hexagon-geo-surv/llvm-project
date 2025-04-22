; RUN: not llvm-as %s --disable-output 2>&1 | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

target datalayout = "A5"

; CHECK: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.1 = alloca i32, align 4, addrspace(1)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.2 = alloca i32, align 4, addrspace(2)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.3 = alloca i32, align 4, addrspace(3)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.4 = alloca i32, align 4, addrspace(4)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.6 = alloca i32, align 4, addrspace(6)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.7 = alloca i32, align 4, addrspace(7)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.8 = alloca i32, align 4, addrspace(8)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(0) or addrspace(5)
; CHECK-NEXT: %alloca.9 = alloca i32, align 4, addrspace(9)
define void @foo() {
entry:
  %alloca.0 = alloca i32, align 4
  %alloca.1 = alloca i32, align 4, addrspace(1)
  %alloca.2 = alloca i32, align 4, addrspace(2)
  %alloca.3 = alloca i32, align 4, addrspace(3)
  %alloca.4 = alloca i32, align 4, addrspace(4)
  %alloca.5 = alloca i32, align 4, addrspace(5)
  %alloca.6 = alloca i32, align 4, addrspace(6)
  %alloca.7 = alloca i32, align 4, addrspace(7)
  %alloca.8 = alloca i32, align 4, addrspace(8)
  %alloca.9 = alloca i32, align 4, addrspace(9)
  ret void
}
