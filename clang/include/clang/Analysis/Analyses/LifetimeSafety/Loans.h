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
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using LoanID = utils::ID<struct LoanTag>;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, LoanID ID) {
  return OS << ID.Value;
}

/// Represents the base of a placeholder access path, which is either a
/// function parameter or the implicit 'this' object of an instance method.
/// Placeholder paths never expire within the function scope, as they represent
/// storage from the caller's scope.
class PlaceholderBase {
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
};

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable or a field within it: var.field.*
///
/// An AccessPath consists of base path which is either a ValueDecl,
/// MaterializeTemporaryExpr, or PlaceholderBase.
///
/// TODO: Model access paths of other types, e.g. field, array subscript, heap
/// and globals.
class AccessPath {
  /// The base of the access path: a variable, temporary, or placeholder.
  const llvm::PointerUnion<const clang::ValueDecl *,
                           const clang::MaterializeTemporaryExpr *,
                           const PlaceholderBase *>
      Base;

public:
  AccessPath(const clang::ValueDecl *D) : Base(D) {}
  AccessPath(const clang::MaterializeTemporaryExpr *MTE) : Base(MTE) {}
  AccessPath(const PlaceholderBase *PB) : Base(PB) {}
  /// Creates an extended access path by appending a path element.
  /// Example: AccessPath(x_path, field) creates path to `x.field`.
  AccessPath(const AccessPath &Other) : Base(Other.Base) {}
  const clang::ValueDecl *getAsValueDecl() const {
    return Base.dyn_cast<const clang::ValueDecl *>();
  }
  const clang::MaterializeTemporaryExpr *getAsMaterializeTemporaryExpr() const {
    return Base.dyn_cast<const clang::MaterializeTemporaryExpr *>();
  }
  const PlaceholderBase *getAsPlaceholderBase() const {
    return Base.dyn_cast<const PlaceholderBase *>();
  }
  bool operator==(const AccessPath &RHS) const { return Base == RHS.Base; }
  bool operator!=(const AccessPath &RHS) const { return !(Base == RHS.Base); }
  void dump(llvm::raw_ostream &OS) const;
};

/// Represents lending a storage location.
//
/// A loan tracks the borrowing relationship created by operations like
/// taking a pointer/reference (&x), creating a view (std::string_view sv = s),
/// or receiving a parameter.
///
/// Examples:
///   - `int* p = &x;` creates a loan to `x`
///   - Parameter loans have no IssueExpr (created at function entry)
class Loan {
  const LoanID ID;
  const AccessPath Path;
  /// The expression that creates the loan, e.g., &x. Null for placeholder
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
public:
  LoanManager() = default;

  Loan *createLoan(AccessPath Path, const Expr *IssueExpr = nullptr) {
    void *Mem = LoanAllocator.Allocate<Loan>();
    auto *NewLoan = new (Mem) Loan(getNextLoanID(), Path, IssueExpr);
    AllLoans.push_back(NewLoan);
    return NewLoan;
  }

  const Loan *getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }

  /// Gets or creates a placeholder base for a given parameter or method.
  const PlaceholderBase *getOrCreatePlaceholderBase(const ParmVarDecl *PVD);
  const PlaceholderBase *getOrCreatePlaceholderBase(const CXXMethodDecl *MD);

  llvm::ArrayRef<const Loan *> getLoans() const { return AllLoans; }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<const Loan *> AllLoans;
  llvm::DenseMap<const Decl *, const PlaceholderBase *> PlaceholderBases;
  llvm::BumpPtrAllocator LoanAllocator;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
