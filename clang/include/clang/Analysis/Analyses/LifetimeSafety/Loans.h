//===- Loans.h - Loan and Access Path Definitions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Loan and AccessPath structures, which represent
// borrows of storage locations, and the LoanManager, which manages the
// creation and retrieval of loans during lifetime analysis.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using LoanID = utils::ID<struct LoanTag>;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, LoanID ID) {
  return OS << ID.Value;
}

/// Represents one step in an access path: either a field access or an
/// interior access (denoted by '*').
class PathElement {
public:
  enum class Kind { Field, Interior };

  static PathElement getField(const FieldDecl *FD) {
    return PathElement(Kind::Field, FD);
  }
  static PathElement getInterior() { return PathElement(Kind::Interior, nullptr); }

  bool isField() const { return K == Kind::Field; }
  bool isInterior() const { return K == Kind::Interior; }
  const FieldDecl *getFieldDecl() const {
    assert(isField());
    return FD;
  }

  bool operator==(const PathElement &Other) const {
    return K == Other.K && FD == Other.FD;
  }
  bool operator<(const PathElement &Other) const {
    if (K != Other.K)
      return static_cast<int>(K) < static_cast<int>(Other.K);
    return FD < Other.FD;
  }
  bool operator!=(const PathElement &Other) const { return !(*this == Other); }

  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddInteger(static_cast<char>(K));
    IDBuilder.AddPointer(FD);
  }

  void dump(llvm::raw_ostream &OS) const {
    if (isField())
      OS << "." << FD->getNameAsString();
    else
      OS << ".*";
  }

private:
  PathElement(Kind K, const FieldDecl *FD) : K(K), FD(FD) {}
  Kind K;
  const FieldDecl *FD;
};

/// Represents the base of a placeholder access path, which is either a
/// function parameter or the implicit 'this' object of an instance method.
class PlaceholderBase : public llvm::FoldingSetNode {
  llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *> ParamOrMethod;

public:
  PlaceholderBase(const ParmVarDecl *PVD) : ParamOrMethod(PVD) {}
  PlaceholderBase(const CXXMethodDecl *MD) : ParamOrMethod(MD) {}

  const ParmVarDecl *getParmVarDecl() const {
    return ParamOrMethod.dyn_cast<const ParmVarDecl *>();
  }

  const CXXMethodDecl *getMethodDecl() const {
    return ParamOrMethod.dyn_cast<const CXXMethodDecl *>();
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(ParamOrMethod.getOpaqueValue());
  }
};

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable or a field within it: var.field.*
/// TODO: Model access paths of other types, e.g. heap and globals.
class AccessPath {
  // An access path can be:
  // - ValueDecl * , to represent the storage location corresponding to the
  //   variable declared in ValueDecl.
  // - MaterializeTemporaryExpr * , to represent the storage location of the
  //   temporary object materialized via this MaterializeTemporaryExpr.
  // - PlaceholderBase * , to represent a borrow from the caller's scope (e.g. a
  //   parameter).
  const llvm::PointerUnion<const clang::ValueDecl *,
                           const clang::MaterializeTemporaryExpr *,
                           const PlaceholderBase *>
      Base;
  llvm::SmallVector<PathElement> Elements;

public:
  AccessPath(const clang::ValueDecl *D) : Base(D) {}
  AccessPath(const clang::MaterializeTemporaryExpr *MTE) : Base(MTE) {}
  AccessPath(const PlaceholderBase *PB) : Base(PB) {}

  AccessPath(const AccessPath &Other, PathElement E)
      : Base(Other.Base), Elements(Other.Elements) {
    Elements.push_back(E);
  }

  const clang::ValueDecl *getAsValueDecl() const {
    return Base.dyn_cast<const clang::ValueDecl *>();
  }

  const clang::MaterializeTemporaryExpr *getAsMaterializeTemporaryExpr() const {
    return Base.dyn_cast<const clang::MaterializeTemporaryExpr *>();
  }

  const PlaceholderBase *getAsPlaceholderBase() const {
    return Base.dyn_cast<const PlaceholderBase *>();
  }

  bool operator==(const AccessPath &RHS) const {
    return Base == RHS.Base && Elements == RHS.Elements;
  }

  /// Returns true if this path is a strict prefix of Other.
  bool isStrictPrefixOf(const AccessPath &Other) const {
    if (Base != Other.Base)
      return false;
    if (Elements.size() >= Other.Elements.size())
      return false;
    for (size_t i = 0; i < Elements.size(); ++i) {
      if (Elements[i] != Other.Elements[i])
        return false;
    }
    return true;
  }

  /// Returns true if this path is a prefix of Other (or same as Other).
  bool isPrefixOf(const AccessPath &Other) const {
    if (Base != Other.Base)
      return false;
    if (Elements.size() > Other.Elements.size())
      return false;
    for (size_t i = 0; i < Elements.size(); ++i) {
      if (Elements[i] != Other.Elements[i])
        return false;
    }
    return true;
  }

  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddPointer(Base.getOpaqueValue());
    for (const auto &E : Elements)
      E.Profile(IDBuilder);
  }

  void dump(llvm::raw_ostream &OS) const;
};

/// Represents lending a storage location.
class Loan {
  const LoanID ID;
  const AccessPath Path;
  /// The expression that creates the loan, e.g., &x. Optional for placeholder
  /// loans.
  const Expr *IssueExpr;

public:
  Loan(LoanID ID, AccessPath Path, const Expr *IssueExpr = nullptr)
      : ID(ID), Path(Path), IssueExpr(IssueExpr) {}

  LoanID getID() const { return ID; }
  const AccessPath &getAccessPath() const { return Path; }
  const Expr *getIssueExpr() const { return IssueExpr; }

  void dump(llvm::raw_ostream &OS) const;
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
  using ExtensionCacheKey = std::pair<LoanID, PathElement>;

public:
  LoanManager() = default;

  Loan *createLoan(AccessPath Path, const Expr *IssueExpr = nullptr) {
    void *Mem = LoanAllocator.Allocate<Loan>();
    auto *NewLoan = new (Mem) Loan(getNextLoanID(), Path, IssueExpr);
    AllLoans.push_back(NewLoan);
    return NewLoan;
  }

  /// Gets or creates a placeholder base for a given parameter or method.
  const PlaceholderBase *getOrCreatePlaceholderBase(const ParmVarDecl *PVD);
  const PlaceholderBase *getOrCreatePlaceholderBase(const CXXMethodDecl *MD);

  /// Gets or creates a loan by extending BaseLoanID with Element.
  /// Caches the result to ensure convergence in LoanPropagation.
  Loan *getOrCreateExtendedLoan(LoanID BaseLoanID, PathElement Element,
                                const Expr *ContextExpr);

  const Loan *getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }
  llvm::ArrayRef<const Loan *> getLoans() const { return AllLoans; }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<const Loan *> AllLoans;
  llvm::BumpPtrAllocator LoanAllocator;
  llvm::FoldingSet<PlaceholderBase> PlaceholderBases;
  std::map<ExtensionCacheKey, Loan *> ExtensionCache;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
