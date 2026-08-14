; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Divergence of phi nodes at divergent cycle exit blocks. A cycle exit phi is a
; join of the paths that leave the cycle and those that stay in it. When threads
; leave the cycle divergently, such a phi is divergent if it selects differing
; values along the exit edges -- even when the incoming values are plain
; constants and nothing is defined inside the cycle (so usesValueFromCycle() and
; the temporal-divergence path do not apply).

declare i32 @llvm.amdgcn.workitem.id.x()

; The cycle is left along two edges that both originate inside it (%body and
; %loop) and the exit phi picks a different constant along each. The branch in
; %body tests the thread id, so threads leave divergently and %acc is divergent.
define amdgpu_kernel void @divergent_cycle_exit_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'divergent_cycle_exit_phi':
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %body ]
  %exit.cond = icmp uge i32 %iv, %n
  br i1 %exit.cond, label %exit, label %body

body:
  %div.cond = icmp eq i32 %tid, 0
  %iv.next = add i32 %iv, 1
  br i1 %div.cond, label %exit, label %loop

exit:
; CHECK: DIVERGENT:   %acc = phi i32 [ 1, %body ], [ 0, %loop ]
  %acc = phi i32 [ 1, %body ], [ 0, %loop ]
  ret void
}

; Same structure, but both in-cycle exit edges carry the same value. Whichever
; edge a thread leaves along it observes the same value, so %acc is uniform.
define amdgpu_kernel void @uniform_cycle_exit_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'uniform_cycle_exit_phi':
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %body ]
  %exit.cond = icmp uge i32 %iv, %n
  br i1 %exit.cond, label %exit, label %body

body:
  %div.cond = icmp eq i32 %tid, 0
  %iv.next = add i32 %iv, 1
  br i1 %div.cond, label %exit, label %loop

exit:
; CHECK-NOT: DIVERGENT:   %acc = phi i32 [ 7, %body ], [ 7, %loop ]
  %acc = phi i32 [ 7, %body ], [ 7, %loop ]
  ret void
}

; Nested cycles. The divergent branch is in the inner cycle (%inner.body) but
; its exit edge leaves both cycles at once. The exit phi merges a value from the
; inner cycle (%inner.body) with a value from the enclosing outer cycle
; (%outer.header). The predecessor carrying the "0" lives in the outer cycle
; only, so the analysis must consider the outermost cycle left by the exit --
; not just the branch's own inner cycle -- to see that %acc is divergent.
define amdgpu_kernel void @nested_cycle_exit_phi() {
; CHECK-LABEL: UniformityInfo for function 'nested_cycle_exit_phi':
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %outer.header

outer.header:
  %o = phi i32 [ 0, %entry ], [ %o.next, %outer.latch ]
  %o.cond = icmp slt i32 %o, 2
  br i1 %o.cond, label %inner.header, label %exit

inner.header:
  %i = phi i32 [ 0, %outer.header ], [ %i.next, %inner.body ]
  %i.cond = icmp slt i32 %i, 2
  br i1 %i.cond, label %inner.body, label %outer.latch

inner.body:
  %i.next = add i32 %i, 1
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit, label %inner.header

outer.latch:
  %o.next = add i32 %o, 1
  br label %outer.header

exit:
; CHECK: DIVERGENT:   %acc = phi i32 [ 1, %inner.body ], [ 0, %outer.header ]
  %acc = phi i32 [ 1, %inner.body ], [ 0, %outer.header ]
  ret void
}

; A cycle with a divergent exit (%body -> %exit.divergent) that also has a second
; exit (%exit.uniform) reached from two in-cycle predecessors (%loop and %mid).
; Both edges into %exit.uniform are governed by uniform branches, so every thread
; that reaches it arrives from the same predecessor and %acc is uniform. Only
; genuinely divergent exits are examined, so the differing constants here must
; not be mistaken for divergence.
define amdgpu_kernel void @uniform_multi_exit_cycle_phi(i32 %n) {
; CHECK-LABEL: UniformityInfo for function 'uniform_multi_exit_cycle_phi':
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %uni.cond = icmp slt i32 %n, 3
  br i1 %uni.cond, label %body, label %exit.uniform

body:
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %exit.divergent, label %mid

mid:
  %uni.cond2 = icmp sgt i32 %n, 1
  br i1 %uni.cond2, label %exit.uniform, label %loop

exit.divergent:
  ret void

exit.uniform:
; CHECK-NOT: DIVERGENT:   %acc = phi i32 [ 0, %loop ], [ 1, %mid ]
  %acc = phi i32 [ 0, %loop ], [ 1, %mid ]
  ret void
}
