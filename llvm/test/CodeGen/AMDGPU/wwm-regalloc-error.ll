; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -stress-regalloc=2 -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -stress-regalloc=2 -stop-after=si-lower-sgpr-spills -o %t.mir %s
; RUN: FileCheck --check-prefix=MIR %s < %t.mir
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -stress-regalloc=2 -start-after=si-lower-sgpr-spills -o - %t.mir | FileCheck --check-prefix=ROUNDTRIP %s

; All allocatable VGPRs are mentioned by inline assembly. Ordinary SGPR spills
; must fall back to scratch memory when no separate WWM pool is available.

; CHECK-LABEL: test:
; CHECK: buffer_store_dword {{.*}} ; 4-byte Folded Spill
; CHECK: buffer_load_dword {{.*}} ; 4-byte Folded Reload
; CHECK: s_endpgm
; CHECK: .amdhsa_private_segment_fixed_size 20

; MIR: hasNoWWMPoolSGPRSpillFallback: true

; ROUNDTRIP-LABEL: test:
; ROUNDTRIP: s_endpgm
; ROUNDTRIP: .amdhsa_private_segment_fixed_size 20

define amdgpu_kernel void @test(i32 %in) {
entry:
  call void asm sideeffect "", "~{v[0:7]}" ()
  call void asm sideeffect "", "~{v[8:15]}" ()
  call void asm sideeffect "", "~{v[16:23]}" ()
  call void asm sideeffect "", "~{v[24:31]}" ()
  call void asm sideeffect "", "~{v[32:39]}" ()
  call void asm sideeffect "", "~{v[40:47]}" ()
  call void asm sideeffect "", "~{v[48:55]}" ()
  call void asm sideeffect "", "~{v[56:63]}" ()
  %val0 = call i32 asm sideeffect "; def $0", "=s" ()
  %val1 = call i32 asm sideeffect "; def $0", "=s" ()
  %val2 = call i32 asm sideeffect "; def $0", "=s" ()
  %cmp = icmp eq i32 %in, 0
  br i1 %cmp, label %bb0, label %ret
bb0:
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  call void asm sideeffect "; use $0", "s"(i32 %val2)
  br label %ret
ret:
  ret void
}
