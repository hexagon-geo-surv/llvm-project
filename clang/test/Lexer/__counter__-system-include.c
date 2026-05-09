// RUN: %clang_cc1                  -Wpedantic %s -fsyntax-only -isystem %S/Inputs -verify=ext
// RUN: %clang_cc1 -std=c2y         -Wpedantic %s -fsyntax-only -isystem %S/Inputs -verify
// RUN: %clang_cc1 -std=c2y -Wpre-c2y-compat   %s -fsyntax-only -isystem %S/Inputs -verify=pre
// RUN: %clang_cc1                            -pedantic %s -fsyntax-only -isystem %S/Inputs -verify=ext
// RUN: %clang_cc1 -std=c2y -Wpre-c2y-compat  -pedantic %s -fsyntax-only -isystem %S/Inputs -verify=pre

#include <__counter__-system-header.h>

// expected-no-diagnostics

int tu_direct_reference = __COUNTER__;
// ext-warning@-1 {{'__COUNTER__' is a C2y extension}}
// pre-warning@-2 {{'__COUNTER__' is incompatible with standards before C2y}}
int tu_counter_alias = COUNTER_ALIAS;
int tu_counter_macro = COUNTER_MACRO();
