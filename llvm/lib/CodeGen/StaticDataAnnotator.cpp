//===- StaticDataAnnotator - Annotate static data's section prefix --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// To reason about module-wide data hotness in a module granularity, this file
// implements a module pass StaticDataAnnotator to work coordinately with the
// StaticDataSplitter pass.
//
// The StaticDataSplitter pass is a machine function pass. It analyzes data
// hotness based on code and adds counters in StaticDataProfileInfo via its
// wrapper pass StaticDataProfileInfoWrapper.
// The StaticDataProfileInfoWrapper sits in the middle between the
// StaticDataSplitter and StaticDataAnnotator passes.
// The StaticDataAnnotator pass is a module pass. It iterates global variables
// in the module, looks up counters from StaticDataProfileInfo and sets the
// section prefix based on profiles.
//
// The three-pass structure is implemented for practical reasons, to work around
// the limitation that a module pass based on legacy pass manager cannot make
// use of MachineBlockFrequencyInfo analysis. In the future, we can consider
// porting the StaticDataSplitter pass to a module-pass using the new pass
// manager framework. That way, analysis are lazily computed as opposed to
// eagerly scheduled, and a module pass can use MachineBlockFrequencyInfo.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

#define DEBUG_TYPE "static-data-annotator"

using namespace llvm;

/// A module pass which iterates global variables in the module and annotates
/// their section prefixes based on profile-driven analysis.
class StaticDataAnnotator : public ModulePass {

public:
  static char ID;

  StaticDataProfileInfo *SDPI = nullptr;
  const ProfileSummaryInfo *PSI = nullptr;

  std::string MemoryProfileFile;
  TargetMachine *TM = nullptr;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;

  std::unique_ptr<IndexedInstrProfReader> MemProfReader;
  // A non-owned pointer to the DataAccessProfData object.
  memprof::DataAccessProfData *DataAccessProfile = nullptr;

  StaticDataAnnotator() : ModulePass(ID), MemoryProfileFile("") {
    initializeStaticDataAnnotatorPass(*PassRegistry::getPassRegistry());
  }

  StaticDataAnnotator(std::string MemoryProfileFile,
                      TargetMachine *TM = nullptr)
      : ModulePass(ID), MemoryProfileFile(MemoryProfileFile), TM(TM) {
    initializeStaticDataAnnotatorPass(*PassRegistry::getPassRegistry());
    // TODO: Pass PGOOptions->FS to the constructor and override this one.
    FS = vfs::getRealFileSystem();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<StaticDataProfileInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Static Data Annotator"; }

  bool runOnModule(Module &M) override;
};

static bool inStaticDataSection(const GlobalVariable &GV,
                                const TargetMachine &TM) {
  SectionKind Kind = TargetLoweringObjectFile::getKindForGlobal(&GV, TM);
  return Kind.isData() || Kind.isReadOnly() || Kind.isReadOnlyWithRel() ||
         Kind.isBSS();
}

bool StaticDataAnnotator::runOnModule(Module &M) {
  SDPI = &getAnalysis<StaticDataProfileInfoWrapperPass>()
              .getStaticDataProfileInfo();
  PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  if (!PSI->hasProfileSummary())
    return false;

  if (!MemoryProfileFile.empty()) {
    auto MemProfReaderOrErr =
        IndexedInstrProfReader::create(MemoryProfileFile, *FS);
    // TODO: improve error handling here like how MemProfUse does it.
    if (Error E = MemProfReaderOrErr.takeError()) {
      LLVM_DEBUG(dbgs() << "Failed to read memory profile file:\n");
    } else {
      MemProfReader = std::move(MemProfReaderOrErr.get());
      DataAccessProfile = MemProfReader->getDataAccessProfileData();
    }
  }

  bool Changed = false;
  for (auto &GV : M.globals()) {
    if (GV.isDeclarationForLinker())
      continue;
    if (!inStaticDataSection(GV, *TM))
      continue;

    // The implementation below assumes prior passes don't set section prefixes,
    // and specifically do 'assign' rather than 'update'. So report error if a
    // section prefix is already set.
    if (auto maybeSectionPrefix = GV.getSectionPrefix();
        maybeSectionPrefix && !maybeSectionPrefix->empty())
      llvm::report_fatal_error("Global variable " + GV.getName() +
                               " already has a section prefix " +
                               *maybeSectionPrefix);
    bool KnownColdSymbol = false;
    if (DataAccessProfile) {
      // If the DataAccessProfile is available, use it to determine if the
      // global variable is a known cold symbol.
      KnownColdSymbol = DataAccessProfile->isKnownColdSymbol(GV.getName());
      auto ProfRecord = DataAccessProfile->getProfileRecord(GV.getName());
      if (ProfRecord) {
        GV.setSectionPrefix("hot");
        Changed = true;
        continue;
      }
    }

    StringRef SectionPrefix = SDPI->getConstantSectionPrefix(&GV, PSI);

    if (SectionPrefix != "hot" && KnownColdSymbol) {
      GV.setSectionPrefix("unlikely");
      continue;
    }

    if (SectionPrefix.empty())
      continue;

    GV.setSectionPrefix(SectionPrefix);
    Changed = true;
  }

  return Changed;
}

char StaticDataAnnotator::ID = 0;

INITIALIZE_PASS(StaticDataAnnotator, DEBUG_TYPE, "Static Data Annotator", false,
                false)

ModulePass *llvm::createStaticDataAnnotatorPass(std::string MemoryProfileFile,
                                                TargetMachine *TM) {
  return new StaticDataAnnotator(MemoryProfileFile, TM);
}
