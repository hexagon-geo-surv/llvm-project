; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize32,-real-true16 < %s | FileCheck -check-prefix=GFX11 %s

; Verify that SIPromoteGlobalLoadSAddr promotes a loop-carried vreg_64 pointer
; phi whose initial value is (sgpr_base + divergent_offset) and whose loop-back
; value advances the base by a uniform stride.
;
; In the loop body the pass must:
;   - select the SADDR form of global_load (vaddr=vgpr_tid, saddr=s[base:base+1])
;   - advance the base with s_add_u32 / s_addc_u32
;   - eliminate the dead vreg_64 phi cycle, leaving no v_add_co in the loop

define amdgpu_kernel void @saddr_loop_carried_half(
    ptr addrspace(1) noalias readonly %in,
    ptr addrspace(1) noalias writeonly %out,
    i32 %K) {
; GFX10-LABEL: saddr_loop_carried_half:
; GFX10:      ; =>This Inner Loop Header:
; GFX10-NOT:    v_add_co_u32
; GFX10:        global_load_ushort {{v[0-9]+}}, {{v[0-9]+}}, s[0:1]
; GFX10:        s_add_u32 s0, s0, 8
; GFX10-NEXT:   s_addc_u32 s1, s1, 0
; GFX10-NOT:    v_add_co_u32
; GFX10:        s_cbranch_scc1
;
; GFX11-LABEL: saddr_loop_carried_half:
; GFX11:      ; =>This Inner Loop Header:
; GFX11-NOT:    v_add_co_u32
; GFX11:        global_load_u16 {{v[0-9]+}}, {{v[0-9]+}}, s[0:1]
; GFX11:        s_add_u32 s0, s0, 8
; GFX11-NEXT:   s_addc_u32 s1, s1, 0
; GFX11-NOT:    v_add_co_u32
; GFX11:        s_cbranch_scc1
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid64 = zext nneg i32 %tid to i64
  %cond = icmp sgt i32 %K, 0
  br i1 %cond, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %base = phi ptr addrspace(1) [ %in, %loop.ph ], [ %base.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %loop.ph ], [ %acc.next, %loop ]
  %gep = getelementptr inbounds [2 x i8], ptr addrspace(1) %base, i64 %tid64
  %val16 = load half, ptr addrspace(1) %gep, align 2
  %val = fpext half %val16 to float
  %acc.next = fadd float %acc, %val
  %base.next = getelementptr inbounds i8, ptr addrspace(1) %base, i64 8
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %K
  br i1 %done, label %exit, label %loop

exit:
  %acc.exit = phi float [ 0.0, %entry ], [ %acc.next, %loop ]
  %out.gep = getelementptr inbounds [4 x i8], ptr addrspace(1) %out, i64 %tid64
  store float %acc.exit, ptr addrspace(1) %out.gep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
