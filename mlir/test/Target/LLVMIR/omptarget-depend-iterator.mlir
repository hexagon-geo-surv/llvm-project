// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test: target construct with iterator-based depend clause.
// The iterator(i=1:10) should allocate a kmp_dep_info[10] array, fill it via
// a dep_iterator loop, then emit __kmpc_omp_wait_deps with ndeps=10 (since
// no nowait, the target task is an included task using begin_if0/complete_if0).

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @omp_target_depend_iterator(%addr: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c10 = llvm.mlir.constant(10 : i64) : i64
    %step = llvm.mlir.constant(1 : i64) : i64

    %it = omp.iterator(%iv: i64) = (%c1 to %c10 step %step) {
      omp.yield(%addr : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>

    %map = omp.map.info var_ptr(%addr : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "data"}
    omp.target depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) map_entries(%map -> %arg0 : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_target_depend_iterator
// CHECK-SAME: (ptr %[[ADDR:[0-9]+]])
// CHECK-DAG: %[[DEP_ARR:.*]] = alloca %struct.kmp_dep_info, i64 10
//
// Iterator loop: preheader -> header -> cond -> body -> inc -> header...
// CHECK: omp_dep_iterator.header:
// CHECK: %[[IV:.*]] = phi i64 [ 0, %omp_dep_iterator.preheader ], [ %[[NEXT:.*]], %omp_dep_iterator.inc ]
// CHECK: omp_dep_iterator.cond:
// CHECK: %[[CMP:.*]] = icmp ult i64 %[[IV]], 10
// CHECK: br i1 %[[CMP]], label %omp_dep_iterator.body, label %omp_dep_iterator.exit
//
// Body: store kmp_dep_info at depArray[0 + linearIV]
// CHECK: omp_dep_iterator.body:
// CHECK: %[[IDX:.*]] = add i64 0, %[[IV]]
// CHECK: %[[ENTRY:.*]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR]], i64 %[[IDX]]
// CHECK: %[[BASE_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 0
// CHECK: %[[PTRINT:.*]] = ptrtoint ptr %[[ADDR]] to i64
// CHECK: store i64 %[[PTRINT]], ptr %[[BASE_GEP]]
// CHECK: %[[LEN_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 1
// CHECK: store i64 8, ptr %[[LEN_GEP]]
// CHECK: %[[FLAGS_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 2
// depKind = 1 (DepIn)
// CHECK: store i8 1, ptr %[[FLAGS_GEP]]
//
// CHECK: omp_dep_iterator.inc:
// CHECK: %[[NEXT]] = add nuw i64 %[[IV]], 1
//
// Target task: wait_deps with ndeps=10, then begin_if0/proxy/complete_if0
// CHECK: call void @__kmpc_omp_wait_deps(ptr @{{.*}}, i32 %{{.*}}, i32 10, ptr %[[DEP_ARR]], i32 0, ptr null)
// CHECK: call void @__kmpc_omp_task_begin_if0
// CHECK: call void @.omp_target_task_proxy_func
// CHECK: call void @__kmpc_omp_task_complete_if0
