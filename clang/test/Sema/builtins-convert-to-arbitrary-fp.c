// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef unsigned _BitInt(8) v4b8 __attribute__((ext_vector_type(4)));

const char *runtime_string;

void test_arity(float x) {
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest");           // expected-error {{too few arguments to function call, expected 4, have 3}}
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest", 0, 0);     // expected-error {{too many arguments to function call, expected 4, have 5}}
}

void test_format(float x) {
  (void)__builtin_convert_to_arbitrary_fp(x, runtime_string, "round.tonearest", 0); // expected-error {{expression is not a string literal}}
  (void)__builtin_convert_to_arbitrary_fp(x, "Nope", "round.tonearest", 0);         // expected-error {{'Nope' is not a supported arbitrary floating-point format}}
}

void test_rounding_mode(float x) {
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", runtime_string, 0);  // expected-error {{expression is not a string literal}}
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.bogus", 0);   // expected-error {{'round.bogus' is not a supported rounding mode}}
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.dynamic", 0); // expected-error {{'round.dynamic' is not a supported rounding mode}}
}

void test_saturate(float x, int i) {
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest", i); // expected-error {{argument to '__builtin_convert_to_arbitrary_fp' must be a constant integer}}
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest", 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}

void test_value(int i, void *p) {
  (void)__builtin_convert_to_arbitrary_fp(i, "Float8E5M2", "round.tonearest", 0); // expected-error {{first argument to __builtin_convert_to_arbitrary_fp must be a floating-point type or a vector of floating-point types}}
  (void)__builtin_convert_to_arbitrary_fp(p, "Float8E5M2", "round.tonearest", 0); // expected-error {{first argument to __builtin_convert_to_arbitrary_fp must be a floating-point type or a vector of floating-point types}}
}

// Clang cannot spell a vector of 6-bit _BitInt, so the six-bit formats have no
// vector form.
void test_vector_format(v4f32 v) {
  (void)__builtin_convert_to_arbitrary_fp(v, "Float6E3M2FN", "round.tonearest", 1); // expected-error {{vector arguments to __builtin_convert_to_arbitrary_fp are not supported for format 'Float6E3M2FN'}}
  (void)__builtin_convert_to_arbitrary_fp(v, "Float6E2M3FN", "round.tonearest", 1); // expected-error {{vector arguments to __builtin_convert_to_arbitrary_fp are not supported for format 'Float6E2M3FN'}}
}

// The result type is derived from the format.
_Static_assert(_Generic(__builtin_convert_to_arbitrary_fp(1.0f, "Float8E5M2", "round.tonearest", 0),
                        unsigned _BitInt(8): 1, default: 0), "");
_Static_assert(_Generic(__builtin_convert_to_arbitrary_fp(1.0f, "Float6E3M2FN", "round.tonearest", 1),
                        unsigned _BitInt(6): 1, default: 0), "");
_Static_assert(_Generic(__builtin_convert_to_arbitrary_fp(1.0f, "Float4E2M1FN", "round.tonearest", 1),
                        unsigned _BitInt(4): 1, default: 0), "");

void test_accepted(float x, v4f32 v) {
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2FNUZ", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E4M3", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E4M3FN", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E4M3FNUZ", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E4M3B11FNUZ", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E3M4", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E8M0FNU", "round.tonearest", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float6E3M2FN", "round.tonearest", 1);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float6E2M3FN", "round.tonearest", 1);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float4E2M1FN", "round.tonearest", 1);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.towardzero", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.upward", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.downward", 0);
  (void)__builtin_convert_to_arbitrary_fp(x, "Float8E5M2", "round.tonearestaway", 0);
  v4b8 r = __builtin_convert_to_arbitrary_fp(v, "Float8E5M2", "round.tonearest", 1);
  (void)r;
}

// The two builtins compose.
float test_roundtrip(float x) {
  return __builtin_convert_from_arbitrary_fp(
      __builtin_convert_to_arbitrary_fp(x, "Float8E4M3FN", "round.tonearest", 1),
      "Float8E4M3FN", float);
}

_Static_assert(__has_builtin(__builtin_convert_to_arbitrary_fp), "");
