// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef _Float16 v2f16 __attribute__((ext_vector_type(2)));

unsigned _BitInt(8) to_fp8_e5m2(float x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest", 0);
}

unsigned _BitInt(8) to_fp8_e4m3fn_saturating(float x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float8E4M3FN", "round.tonearest", 1);
}

unsigned _BitInt(8) to_fp8_from_f16(_Float16 x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.towardzero", 0);
}

unsigned _BitInt(8) to_fp8_from_bf16(__bf16 x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.upward", 0);
}

unsigned _BitInt(8) to_fp8_from_f64(double x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.downward", 0);
}

unsigned _BitInt(6) to_fp6_e3m2fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float6E3M2FN", "round.tonearestaway", 1);
}

unsigned _BitInt(6) to_fp6_e2m3fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float6E2M3FN", "round.tonearest", 1);
}

unsigned _BitInt(4) to_fp4_e2m1fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float4E2M1FN", "round.tonearest", 1);
}

typedef unsigned _BitInt(8) v4b8 __attribute__((ext_vector_type(4)));
v4b8 to_fp8_v4(v4f32 x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest", 1);
}

typedef unsigned _BitInt(4) v2b4 __attribute__((ext_vector_type(2)));
v2b4 to_fp4_v2(v2f16 x) {
  return __builtin_convert_to_arbitrary_fp(x, "Float4E2M1FN", "round.tonearest", 1);
}

// CHECK-LABEL: @to_fp8_e5m2(
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %{{.*}}, metadata !"Float8E5M2", metadata !"round.tonearest", i1 false)
//
// CHECK-LABEL: @to_fp8_e4m3fn_saturating(
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %{{.*}}, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 true)
//
// CHECK-LABEL: @to_fp8_from_f16(
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f16(half %{{.*}}, metadata !"Float8E5M2", metadata !"round.towardzero", i1 false)
//
// CHECK-LABEL: @to_fp8_from_bf16(
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.bf16(bfloat %{{.*}}, metadata !"Float8E5M2", metadata !"round.upward", i1 false)
//
// CHECK-LABEL: @to_fp8_from_f64(
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f64(double %{{.*}}, metadata !"Float8E5M2", metadata !"round.downward", i1 false)
//
// CHECK-LABEL: @to_fp6_e3m2fn(
// CHECK: call i6 @llvm.convert.to.arbitrary.fp.i6.f32(float %{{.*}}, metadata !"Float6E3M2FN", metadata !"round.tonearestaway", i1 true)
//
// CHECK-LABEL: @to_fp6_e2m3fn(
// CHECK: call i6 @llvm.convert.to.arbitrary.fp.i6.f32(float %{{.*}}, metadata !"Float6E2M3FN", metadata !"round.tonearest", i1 true)
//
// CHECK-LABEL: @to_fp4_e2m1fn(
// CHECK: call i4 @llvm.convert.to.arbitrary.fp.i4.f32(float %{{.*}}, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 true)
//
// CHECK-LABEL: @to_fp8_v4(
// CHECK: call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f32(<4 x float> %{{.*}}, metadata !"Float8E5M2", metadata !"round.tonearest", i1 true)
//
// CHECK-LABEL: @to_fp4_v2(
// CHECK: call <2 x i4> @llvm.convert.to.arbitrary.fp.v2i4.v2f16(<2 x half> %{{.*}}, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 true)
