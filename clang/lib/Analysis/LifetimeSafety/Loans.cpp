//===- Loans.cpp - Loan Implementation --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"

namespace clang::lifetimes::internal {

void AccessPath::dump(llvm::raw_ostream &OS) const {
  if (const clang::ValueDecl *VD = getAsValueDecl())
    OS << VD->getNameAsString();
  else if (const clang::MaterializeTemporaryExpr *MTE =
               getAsMaterializeTemporaryExpr())
    OS << "MaterializeTemporaryExpr at " << MTE;
  else if (const PlaceholderBase *PB = getAsPlaceholderBase()) {
    if (const auto *PVD = PB->getParmVarDecl())
      OS << "$" << PVD->getNameAsString();
    else if (PB->getMethodDecl())
      OS << "$this";
  } else
    llvm_unreachable("access path base invalid");
}

void Loan::dump(llvm::raw_ostream &OS) const {
  OS << getID() << " (Path: ";
  Path.dump(OS);
  OS << ")";
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const ParmVarDecl *PVD) {
  if (auto It = PlaceholderBases.find(PVD); It != PlaceholderBases.end())
    return It->second;
  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(PVD);
  PlaceholderBases.insert({PVD, NewPB});
  return NewPB;
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const CXXMethodDecl *MD) {
  if (auto It = PlaceholderBases.find(MD); It != PlaceholderBases.end())
    return It->second;
  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(MD);
  PlaceholderBases.insert({MD, NewPB});
  return NewPB;
}
} // namespace clang::lifetimes::internal
