! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Tests for the iterator modifier on the depend clause across all directives
! that Flang currently supports: task, target, target enter data, target exit data,
! and target update.
! TODO: We need to add iterator test for taskwait, depobj, interop once they are
! supported.

!===============================================================================
! task
!===============================================================================

subroutine task_depend_iterator_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp task depend(iterator(i = 1:n), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {
! CHECK:   omp.terminator
! CHECK: }

subroutine task_depend_iterator_2d()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp task depend(iterator(i = 1:n, j = 1:m), inout: a(i, j))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_2d()
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[IV0_I64:.*]] = fir.convert %[[IV0_I32]] : (i32) -> i64
! CHECK:   %[[IV1_I64:.*]] = fir.convert %[[IV1_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c4, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR:.*]] = fir.array_coor %{{.*}}(%[[SHAPE]]) %[[IV0_I64]], %[[IV1_I64]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shape<2>, i64, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependinout -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

subroutine task_depend_iterator_mixed()
  integer, parameter :: n = 16
  integer :: a(n), x
  integer :: i

  !$omp task depend(iterator(i = 1:n), in: a(i)) depend(out: x)
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_mixed()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_mixedEa"}
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtask_depend_iterator_mixedEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependout -> %[[X]]#0 : !fir.ref<i32>, taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

subroutine task_depend_iterator_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp task depend(iterator(i = 1:n:2), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_step()
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[C2_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

!===============================================================================
! target
!===============================================================================

subroutine target_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target depend(iterator(i = 1:n), in: a(i)) map(tofrom: a)
    a(1) = 10
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] -> %{{.*}} : !fir.ref<!fir.array<16xi32>>) {
! CHECK:   omp.terminator
! CHECK: }

!===============================================================================
! target enter data
!===============================================================================

subroutine target_enter_data_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target enter data map(to: a) depend(iterator(i = 1:n), in: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_enter_data depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)

!===============================================================================
! target exit data
!===============================================================================

subroutine target_exit_data_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target exit data map(from: a) depend(iterator(i = 1:n), out: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_exit_data depend(taskdependout -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)

!===============================================================================
! target update
!===============================================================================

subroutine target_update_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(a) depend(iterator(i = 1:n), in: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_update depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)
