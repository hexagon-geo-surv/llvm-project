; RUN: llc -mtriple=amdgcn -mcpu=gfx600 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx700 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx801 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}scalar_andn2_i32_one_use
; GCN: s_andn2_b32
define amdgpu_kernel void @scalar_andn2_i32_one_use(
    ptr addrspace(1) %r0, i32 %a, i32 %b) {
entry:
  %nb = xor i32 %b, -1
  %r0.val = and i32 %a, %nb
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_andn2_i64_one_use
; GCN: s_andn2_b64
define amdgpu_kernel void @scalar_andn2_i64_one_use(
    ptr addrspace(1) %r0, i64 %a, i64 %b) {
entry:
  %nb = xor i64 %b, -1
  %r0.val = and i64 %a, %nb
  store i64 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_orn2_i32_one_use
; GCN: s_orn2_b32
define amdgpu_kernel void @scalar_orn2_i32_one_use(
    ptr addrspace(1) %r0, i32 %a, i32 %b) {
entry:
  %nb = xor i32 %b, -1
  %r0.val = or i32 %a, %nb
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_orn2_i64_one_use
; GCN: s_orn2_b64
define amdgpu_kernel void @scalar_orn2_i64_one_use(
    ptr addrspace(1) %r0, i64 %a, i64 %b) {
entry:
  %nb = xor i64 %b, -1
  %r0.val = or i64 %a, %nb
  store i64 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_andn2_i32_s_v_one_use
; GCN: v_bfi_b32
define amdgpu_kernel void @vector_andn2_i32_s_v_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %v, -1
  %r0.val = and i32 %s, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_andn2_i32_v_s_one_use
; GCN: v_bfi_b32
define amdgpu_kernel void @vector_andn2_i32_v_s_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %s, -1
  %r0.val = and i32 %v, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_orn2_i32_s_v_one_use
; GCN: v_bfi_b32
define amdgpu_kernel void @vector_orn2_i32_s_v_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %v, -1
  %r0.val = or i32 %s, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_orn2_i32_v_s_one_use
; GCN: v_bfi_b32
define amdgpu_kernel void @vector_orn2_i32_v_s_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %s, -1
  %r0.val = or i32 %v, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_bfi_v2i32
; GCN: v_bfi_b32
; GCN: v_bfi_b32
define <2 x i32> @vector_bfi_v2i32(<2 x i32> %x, <2 x i32> %y, <2 x i32> %z) {
entry:
  %ny = and <2 x i32> %y, %x
  %nx = xor <2 x i32> %x, <i32 -1, i32 -1>
  %nz = and <2 x i32> %z, %nx
  %r = or <2 x i32> %ny, %nz
  ret <2 x i32> %r
}

; GCN-LABEL: {{^}}vector_andn2_v2i32_one_use
; GCN: v_bfi_b32
; GCN: v_bfi_b32
define <2 x i32> @vector_andn2_v2i32_one_use(<2 x i32> %v, <2 x i32> %s) {
entry:
  %not = xor <2 x i32> %v, <i32 -1, i32 -1>
  %r = and <2 x i32> %s, %not
  ret <2 x i32> %r
}

; GCN-LABEL: {{^}}vector_orn2_v2i32_one_use
; GCN: v_bfi_b32
; GCN: v_bfi_b32
define <2 x i32> @vector_orn2_v2i32_one_use(<2 x i32> %v, <2 x i32> %s) {
entry:
  %not = xor <2 x i32> %v, <i32 -1, i32 -1>
  %r = or <2 x i32> %s, %not
  ret <2 x i32> %r
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #0
