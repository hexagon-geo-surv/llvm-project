/// SIPromoteGlobalLoadToSAddr - Pre-RA peephole that promotes
///   GLOBAL_LOAD vdst, %addr:vreg_64, off
/// to
///   GLOBAL_LOAD_SADDR vdst, %sbase:sreg_64, %voff32:vgpr_32, off
///
/// Triggered when %addr is a loop-carried vreg_64 phi whose:
///   - preheader incoming value is REG_SEQUENCE(
///       V_ADD_CO_U32(sgpr_base_lo, vgpr_voff32) -> lo,
///       V_ADDC_U32  (sgpr_base_hi, 0, carry)    -> hi)
///   - loop-back value is REG_SEQUENCE(
///       V_ADD_CO_U32(phi.sub0, stride_imm) -> lo,
///       V_ADDC_U32  (phi.sub1, 0, carry)   -> hi)
///
/// This recovers the SADDR addressing mode for the pattern:
///   for (int i = 0; i < K; ++i) acc += base[threadIdx.x]; base += stride;
/// The compiler currently sinks the uniform base pointer into VGPRs instead of
/// keeping it in SGPRs (SADDR) and using the lane offset (VADDR) separately.

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {
void initializeMachineLoopInfoWrapperPassPass(PassRegistry &);
} // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "si-promote-global-load-saddr"

// Map a non-SADDR GLOBAL_LOAD opcode to its SADDR variant.
static unsigned getGlobalSAddrOpcode(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::GLOBAL_LOAD_SHORT_D16_t16:    return AMDGPU::GLOBAL_LOAD_SHORT_D16_SADDR_t16;
  case AMDGPU::GLOBAL_LOAD_SHORT_D16:        return AMDGPU::GLOBAL_LOAD_SHORT_D16_SADDR;
  case AMDGPU::GLOBAL_LOAD_SBYTE_D16_t16:   return AMDGPU::GLOBAL_LOAD_SBYTE_D16_SADDR_t16;
  case AMDGPU::GLOBAL_LOAD_SBYTE_D16:       return AMDGPU::GLOBAL_LOAD_SBYTE_D16_SADDR;
  case AMDGPU::GLOBAL_LOAD_UBYTE_D16_t16:   return AMDGPU::GLOBAL_LOAD_UBYTE_D16_SADDR_t16;
  case AMDGPU::GLOBAL_LOAD_UBYTE_D16:       return AMDGPU::GLOBAL_LOAD_UBYTE_D16_SADDR;
  case AMDGPU::GLOBAL_LOAD_UBYTE:           return AMDGPU::GLOBAL_LOAD_UBYTE_SADDR;
  case AMDGPU::GLOBAL_LOAD_SBYTE:           return AMDGPU::GLOBAL_LOAD_SBYTE_SADDR;
  case AMDGPU::GLOBAL_LOAD_USHORT:          return AMDGPU::GLOBAL_LOAD_USHORT_SADDR;
  case AMDGPU::GLOBAL_LOAD_SSHORT:          return AMDGPU::GLOBAL_LOAD_SSHORT_SADDR;
  case AMDGPU::GLOBAL_LOAD_DWORD:           return AMDGPU::GLOBAL_LOAD_DWORD_SADDR;
  case AMDGPU::GLOBAL_LOAD_DWORDX2:         return AMDGPU::GLOBAL_LOAD_DWORDX2_SADDR;
  case AMDGPU::GLOBAL_LOAD_DWORDX3:         return AMDGPU::GLOBAL_LOAD_DWORDX3_SADDR;
  case AMDGPU::GLOBAL_LOAD_DWORDX4:         return AMDGPU::GLOBAL_LOAD_DWORDX4_SADDR;
  default:                                   return ~0u;
  }
}

namespace {

// All information extracted from the phi pattern needed for the transform.
struct PhiInfo {
  MachineInstr *Phi       = nullptr; // the vreg_64 PHI
  MachineInstr *AdvSeq    = nullptr; // REG_SEQUENCE that produces the loop-back value
  MachineInstr *PreSeq    = nullptr; // REG_SEQUENCE that produces the preheader value
  // Preheader: addr_init = sgpr_base_lo:sgpr_base_hi + voff32 (zext to 64)
  Register SBaseLo;       // SGPR source of lo-half (may be a parent reg with SubLo)
  unsigned SBaseLoSub = 0; // sub-register index for SBaseLo (0 = no subreg)
  Register SBaseHi;       // SGPR source of hi-half (may be a parent reg with SubHi)
  unsigned SBaseHiSub = 0; // sub-register index for SBaseHi (0 = no subreg)
  Register VOff32;    // VGPR32: the divergent lane offset (e.g. threadIdx.x * sizeof(T))
  // Loop advance: addr_next = addr_phi + stride  (stride fits in 32-bit imm, hi=0)
  int64_t  Stride = 0;
  MachineBasicBlock *Preheader = nullptr;
  MachineBasicBlock *LoopBlock = nullptr;
};

class SIPromoteGlobalLoadSAddr : public MachineFunctionPass {
public:
  static char ID;
  SIPromoteGlobalLoadSAddr() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "SI Promote Global Load to SADDR";
  }
};

} // anonymous namespace

char SIPromoteGlobalLoadSAddr::ID = 0;

INITIALIZE_PASS_BEGIN(SIPromoteGlobalLoadSAddr, DEBUG_TYPE,
                      "SI Promote Global Load to SADDR", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SIPromoteGlobalLoadSAddr, DEBUG_TYPE,
                    "SI Promote Global Load to SADDR", false, false)

char &llvm::SIPromoteGlobalLoadSAddrID = SIPromoteGlobalLoadSAddr::ID;

FunctionPass *llvm::createSIPromoteGlobalLoadSAddrPass() {
  return new SIPromoteGlobalLoadSAddr();
}

static bool isVGPR32(Register Reg, MachineRegisterInfo &MRI) {
  return AMDGPU::VGPR_32RegClass.hasSubClassEq(MRI.getRegClass(Reg));
}

// Match a REG_SEQUENCE that packs two vgpr_32 halves into a vreg_64.
// Returns true and fills Lo/Hi.
static bool matchSeq64(MachineInstr *MI, Register &Lo, Register &Hi) {
  if (!MI || MI->getOpcode() != AMDGPU::REG_SEQUENCE)
    return false;
  if (MI->getNumOperands() != 5)
    return false;
  if (!MI->getOperand(2).isImm() || MI->getOperand(2).getImm() != AMDGPU::sub0)
    return false;
  if (!MI->getOperand(4).isImm() || MI->getOperand(4).getImm() != AMDGPU::sub1)
    return false;
  Lo = MI->getOperand(1).getReg();
  Hi = MI->getOperand(3).getReg();
  return true;
}

// Match the preheader incoming value.
//
// Pattern (after SIFoldOperands expands V_ADD_U64_PSEUDO):
//   %lo, %carry = V_ADD_CO_U32_e64  sgpr_base_lo, vgpr_voff32, 0
//   %hi, dead _ = V_ADDC_U32_e64    sgpr_base_hi, 0, killed %carry, 0
//   %seq:vreg_64 = REG_SEQUENCE %lo, sub0, %hi, sub1
//
// sgpr_base_lo/hi are sub-register reads of an SGPR block (kernel arg).
// vgpr_voff32 is the divergent 32-bit lane offset.
static bool matchPreheaderInit(Register InitReg, MachineRegisterInfo &MRI,
                               PhiInfo &Info) {
  MachineInstr *Seq = MRI.getVRegDef(InitReg);
  Register Lo, Hi;
  if (!matchSeq64(Seq, Lo, Hi))
    return false;
  if (!isVGPR32(Lo, MRI) || !isVGPR32(Hi, MRI))
    return false;

  MachineInstr *AddLo = MRI.getVRegDef(Lo);
  if (!AddLo || AddLo->getOpcode() != AMDGPU::V_ADD_CO_U32_e64)
    return false;
  // V_ADD_CO_U32_e64: 0=dst_lo, 1=dst_carry, 2=src0, 3=src1, 4=clamp
  MachineOperand &Src0 = AddLo->getOperand(2);
  MachineOperand &Src1 = AddLo->getOperand(3);
  if (!Src0.isReg() || !Src1.isReg())
    return false;

  MachineInstr *AddHi = MRI.getVRegDef(Hi);
  if (!AddHi || AddHi->getOpcode() != AMDGPU::V_ADDC_U32_e64)
    return false;
  // V_ADDC_U32_e64: 0=dst_hi, 1=dst_carry, 2=src0(sgpr_hi), 3=src1(0), 4=carry_in, 5=clamp
  if (!AddHi->getOperand(3).isImm() || AddHi->getOperand(3).getImm() != 0)
    return false;
  // carry-in of AddHi must be the carry-out of AddLo
  if (!AddHi->getOperand(4).isReg())
    return false;
  Register CarryIn = AddHi->getOperand(4).getReg();
  if (!CarryIn.isVirtual() || MRI.getVRegDef(CarryIn) != AddLo)
    return false;
  MachineOperand &HiSrc0 = AddHi->getOperand(2);
  if (!HiSrc0.isReg() || isVGPR32(HiSrc0.getReg(), MRI))
    return false;

  // One of Src0/Src1 of AddLo is the SGPR base lo, the other is the VGPR voff32.
  auto tryOrder = [&](MachineOperand &MaybeSGPR, MachineOperand &MaybeVGPR) {
    if (isVGPR32(MaybeSGPR.getReg(), MRI))
      return false; // wrong order
    if (!isVGPR32(MaybeVGPR.getReg(), MRI))
      return false;
    Info.SBaseLo    = MaybeSGPR.getReg();
    Info.SBaseLoSub = MaybeSGPR.getSubReg();
    Info.SBaseHi    = HiSrc0.getReg();
    Info.SBaseHiSub = HiSrc0.getSubReg();
    Info.VOff32     = MaybeVGPR.getReg();
    Info.PreSeq  = Seq;
    return true;
  };
  return tryOrder(Src0, Src1) || tryOrder(Src1, Src0);
}

// Match the loop-back advance of the vreg_64 phi.
//
// Pattern (after SIFoldOperands expands V_ADD_U64_PSEUDO):
//   %lo, %carry = V_ADD_CO_U32_e64  %phi.sub0, stride_imm, 0
//   %hi, dead _ = V_ADDC_U32_e64    %phi.sub1, 0, killed %carry, 0
//   %next:vreg_64 = REG_SEQUENCE %lo, sub0, %hi, sub1
//
// stride_imm must be a constant that fits in a 32-bit immediate (hi stride = 0).
static bool matchLoopAdvance(Register NextReg, Register PhiReg,
                             MachineRegisterInfo &MRI, PhiInfo &Info) {
  MachineInstr *Seq = MRI.getVRegDef(NextReg);
  Register Lo, Hi;
  if (!matchSeq64(Seq, Lo, Hi))
    return false;
  if (!isVGPR32(Lo, MRI) || !isVGPR32(Hi, MRI))
    return false;

  MachineInstr *AddLo = MRI.getVRegDef(Lo);
  if (!AddLo || AddLo->getOpcode() != AMDGPU::V_ADD_CO_U32_e64)
    return false;

  MachineInstr *AddHi = MRI.getVRegDef(Hi);
  if (!AddHi || AddHi->getOpcode() != AMDGPU::V_ADDC_U32_e64)
    return false;
  if (!AddHi->getOperand(3).isImm() || AddHi->getOperand(3).getImm() != 0)
    return false;
  if (!AddHi->getOperand(4).isReg())
    return false;
  Register CarryIn = AddHi->getOperand(4).getReg();
  if (!CarryIn.isVirtual() || MRI.getVRegDef(CarryIn) != AddLo)
    return false;

  // AddHi src0 must be phi.sub1.
  MachineOperand &HiSrc0 = AddHi->getOperand(2);
  if (!HiSrc0.isReg() || HiSrc0.getReg() != PhiReg ||
      HiSrc0.getSubReg() != AMDGPU::sub1)
    return false;

  // One of AddLo src0/src1 must be phi.sub0, the other an immediate (stride).
  auto tryOrder = [&](MachineOperand &MaybePhiOp, MachineOperand &MaybeImm) {
    if (!MaybePhiOp.isReg() || MaybePhiOp.getReg() != PhiReg)
      return false;
    if (MaybePhiOp.getSubReg() != AMDGPU::sub0)
      return false;
    if (!MaybeImm.isImm())
      return false;
    Info.Stride  = MaybeImm.getImm();
    Info.AdvSeq  = Seq;
    return true;
  };
  return tryOrder(AddLo->getOperand(2), AddLo->getOperand(3)) ||
         tryOrder(AddLo->getOperand(3), AddLo->getOperand(2));
}

static bool analyzeAddrPhi(Register AddrReg, MachineRegisterInfo &MRI,
                            const MachineLoopInfo &MLI, PhiInfo &Info) {
  MachineInstr *PhiDef = MRI.getVRegDef(AddrReg);
  if (!PhiDef || !PhiDef->isPHI())
    return false;

  MachineBasicBlock *LoopBlock = PhiDef->getParent();
  MachineLoop *Loop = MLI.getLoopFor(LoopBlock);
  if (!Loop)
    return false;

  MachineBasicBlock *Preheader = Loop->getLoopPreheader();
  if (!Preheader)
    return false;

  // PHI must have exactly two incoming edges: preheader and loop-back.
  if (PhiDef->getNumOperands() != 5) // dst, val0, bb0, val1, bb1
    return false;

  Register Val0 = PhiDef->getOperand(1).getReg();
  MachineBasicBlock *BB0 = PhiDef->getOperand(2).getMBB();
  Register Val1 = PhiDef->getOperand(3).getReg();
  MachineBasicBlock *BB1 = PhiDef->getOperand(4).getMBB();

  Register InitVal, NextVal;
  if (BB0 == Preheader) { InitVal = Val0; NextVal = Val1; }
  else if (BB1 == Preheader) { InitVal = Val1; NextVal = Val0; }
  else return false;

  if (!matchPreheaderInit(InitVal, MRI, Info))
    return false;
  if (!matchLoopAdvance(NextVal, AddrReg, MRI, Info))
    return false;

  // The phi must only be used by the load and the advance REG_SEQUENCE
  // (and the two V_ADD_CO/V_ADDC instructions that feed it via sub-regs).
  // Check that no other instruction reads AddrReg directly.
  for (auto &Use : MRI.use_nodbg_instructions(AddrReg)) {
    if (&Use == Info.AdvSeq)
      continue;
    // The V_ADD_CO_U32 and V_ADDC_U32 of the advance read sub-registers of AddrReg.
    unsigned Opc = Use.getOpcode();
    if (Opc == AMDGPU::V_ADD_CO_U32_e64 || Opc == AMDGPU::V_ADDC_U32_e64)
      continue;
    // The load itself — allowed.
    if (getGlobalSAddrOpcode(Use.getOpcode()) != ~0u)
      continue;
    // Anything else: bail.
    return false;
  }

  Info.Phi       = PhiDef;
  Info.Preheader = Preheader;
  Info.LoopBlock = LoopBlock;
  return true;
}

// Perform the transformation for one GLOBAL_LOAD instruction.
//
// Before (loop):
//   %addr:vreg_64  = PHI [preheader_seq, preheader], [adv_seq, loop]
//   GLOBAL_LOAD    vdst, %addr, imm_off, cpol
//   adv_seq:       V_ADD_CO %addr.sub0, stride → lo; V_ADDC %addr.sub1 → hi; REGSEQ
//
// After (loop):
//   %sbase:sreg_64 = PHI [sgpr_base_init, preheader], [%snext, loop]
//   GLOBAL_LOAD_SADDR vdst, %sbase, %voff32, imm_off, cpol
//   %snext_lo = S_ADD_U32  %sbase.sub0, stride
//   %snext_hi = S_ADDC_U32 %sbase.sub1, 0
//   %snext:sreg_64 = REG_SEQUENCE %snext_lo, sub0, %snext_hi, sub1
static bool promote(MachineInstr &Load, const PhiInfo &Info,
                    MachineRegisterInfo &MRI, const SIInstrInfo &TII) {
  unsigned SAddrOpc = getGlobalSAddrOpcode(Load.getOpcode());
  if (SAddrOpc == ~0u)
    return false;

  MachineBasicBlock &LoopMBB = *Info.LoopBlock;
  const DebugLoc &DL = Load.getDebugLoc();

  // --- Create the scalar base phi ---
  // Build a preheader sgpr_64 from the two halves identified in matchPreheaderInit.
  // We need the sgpr_64 for the phi's preheader incoming value.
  // SBaseLo/SBaseHi are the SGPR sub-register sources from the preheader V_ADD_CO.
  // Build a REG_SEQUENCE to pack them into a virtual sreg_64.
  Register SGPRBase = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  Register SGPRNext = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);

  // Insert sgpr_base = REG_SEQUENCE(SBaseLo[.SubLo], sub0, SBaseHi[.SubHi], sub1) in preheader.
  // SBaseLo/Hi may be sub-register reads of a wider SGPR (e.g. sgpr_128 kernel arg).
  MachineBasicBlock::iterator PreEnd = Info.Preheader->getFirstTerminator();
  BuildMI(*Info.Preheader, PreEnd, DL, TII.get(AMDGPU::REG_SEQUENCE), SGPRBase)
      .addReg(Info.SBaseLo, RegState::NoFlags, Info.SBaseLoSub)
      .addImm(AMDGPU::sub0)
      .addReg(Info.SBaseHi, RegState::NoFlags, Info.SBaseHiSub)
      .addImm(AMDGPU::sub1);

  // Insert scalar phi at the top of the loop block.
  MachineBasicBlock *LoopBackMBB = Info.AdvSeq->getParent();
  MachineBasicBlock::iterator PhiInsert = LoopMBB.begin();
  BuildMI(LoopMBB, PhiInsert, DL, TII.get(AMDGPU::PHI), SGPRNext)
      .addReg(SGPRBase).addMBB(Info.Preheader)
      .addReg(SGPRNext).addMBB(LoopBackMBB);

  // --- Replace the load with GLOBAL_LOAD_SADDR ---
  // Non-SADDR layout: vdst, vaddr(vreg_64), offset, cpol
  // SADDR layout:     vdst, saddr(sreg_64), vaddr(vgpr_32), offset, cpol
  int64_t Offset = Load.getOperand(2).getImm();
  int64_t CPol   = Load.getOperand(3).getImm();
  Register OldVdst = Load.getOperand(0).getReg();

  MachineBasicBlock::iterator LoadIt(Load);
  BuildMI(LoopMBB, LoadIt, DL, TII.get(SAddrOpc), OldVdst)
      .addReg(SGPRNext)      // saddr
      .addReg(Info.VOff32)   // vaddr (32-bit lane offset)
      .addImm(Offset)
      .addImm(CPol)
      .cloneMemRefs(Load);
  Load.eraseFromParent();

  // --- Replace the advance REG_SEQUENCE with scalar adds ---
  // AdvSeq: REG_SEQUENCE(%adv_lo, sub0, %adv_hi, sub1)
  // %adv_lo = V_ADD_CO_U32(phi.sub0, stride_imm)
  // %adv_hi = V_ADDC_U32 (phi.sub1, 0, carry)
  // Replace with:
  //   %snext_lo = S_ADD_U32  SGPRNext.sub0, stride_lo
  //   %snext_hi = S_ADDC_U32 SGPRNext.sub1, 0
  //   SGPRNext  = REG_SEQUENCE %snext_lo, sub0, %snext_hi, sub1  [via phi update]
  //
  // We reuse SGPRNext as the phi destination and feed it from a new REG_SEQUENCE.
  // Fix the phi: change its loop-back incoming to the new SGPRNext_seq below.
  Register SNLo = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  Register SNHi = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  Register SNSeq = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);

  MachineBasicBlock::iterator AdvIt(Info.AdvSeq);
  MachineBasicBlock &AdvMBB = *Info.AdvSeq->getParent();
  const DebugLoc &AdvDL = Info.AdvSeq->getDebugLoc();

  int32_t StrLo = static_cast<int32_t>(Info.Stride & 0xFFFFFFFFLL);

  Register PhiSubLo = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  Register PhiSubHi = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  BuildMI(AdvMBB, AdvIt, AdvDL, TII.get(AMDGPU::COPY), PhiSubLo)
      .addReg(SGPRNext, RegState::NoFlags, AMDGPU::sub0);
  BuildMI(AdvMBB, AdvIt, AdvDL, TII.get(AMDGPU::COPY), PhiSubHi)
      .addReg(SGPRNext, RegState::NoFlags, AMDGPU::sub1);
  BuildMI(AdvMBB, AdvIt, AdvDL, TII.get(AMDGPU::S_ADD_U32), SNLo)
      .addReg(PhiSubLo).addImm(StrLo);
  BuildMI(AdvMBB, AdvIt, AdvDL, TII.get(AMDGPU::S_ADDC_U32), SNHi)
      .addReg(PhiSubHi).addImm(0);
  BuildMI(AdvMBB, AdvIt, AdvDL, TII.get(AMDGPU::REG_SEQUENCE), SNSeq)
      .addReg(SNLo).addImm(AMDGPU::sub0)
      .addReg(SNHi).addImm(AMDGPU::sub1);

  // Patch the phi: replace SGPRNext (currently self-referential) with SNSeq
  // for the loop-back edge.
  for (MachineInstr &MI : LoopMBB) {
    if (!MI.isPHI())
      break;
    if (MI.getOperand(0).getReg() != SGPRNext)
      continue;
    for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
      if (MI.getOperand(i).getReg() == SGPRNext &&
          MI.getOperand(i + 1).getMBB() == LoopBackMBB) {
        MI.getOperand(i).setReg(SNSeq);
      }
    }
    break;
  }

  // Remove the dead vreg_64 cycle: phi → advance REG_SEQUENCE → phi.
  // The old phi (%7) is read only by V_ADD_CO_U32/V_ADDC_U32 (via sub-regs),
  // which feed the advance REG_SEQUENCE (%10), which feeds back into the phi.
  // Standard DeadMachineInstructionElim won't break self-referential cycles,
  // so we delete them explicitly in dependency order.
  //
  // Delete: AdvSeq (%10 = REG_SEQUENCE), V_ADD_CO (%50), V_ADDC (%51)
  //         PreSeq (%3  = REG_SEQUENCE), V_ADD_CO (%42), V_ADDC (%43)
  //         old Phi (%7)
  auto killInsn = [](MachineInstr *MI) {
    if (MI && MI->getParent())
      MI->eraseFromParent();
  };

  // Collect the instructions to delete before any erasure invalidates them.
  // AdvSeq feeds into the phi loop-back; its inputs are V_ADD_CO / V_ADDC.
  MachineInstr *AdvSeq  = Info.AdvSeq;
  Register AdvLo = AdvSeq->getOperand(1).getReg(); // sub0 of old advance
  Register AdvHi = AdvSeq->getOperand(3).getReg(); // sub1 of old advance
  MachineInstr *AdvAddLo = MRI.getVRegDef(AdvLo);
  MachineInstr *AdvAddHi = MRI.getVRegDef(AdvHi);

  // PreSeq feeds into the phi's preheader incoming value.
  MachineInstr *PreSeq  = Info.PreSeq;
  Register PreLo = PreSeq ? PreSeq->getOperand(1).getReg() : Register();
  Register PreHi = PreSeq ? PreSeq->getOperand(3).getReg() : Register();
  MachineInstr *PreAddLo = PreLo.isValid() ? MRI.getVRegDef(PreLo) : nullptr;
  MachineInstr *PreAddHi = PreHi.isValid() ? MRI.getVRegDef(PreHi) : nullptr;

  // Delete in reverse-use order so no instruction outlives its defs.
  killInsn(AdvSeq);
  killInsn(AdvAddHi);
  killInsn(AdvAddLo);
  killInsn(Info.Phi);
  killInsn(PreSeq);
  killInsn(PreAddHi);
  killInsn(PreAddLo);

  return true;
}

bool SIPromoteGlobalLoadSAddr::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasFlatGlobalInsts())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  const MachineLoopInfo &MLI =
      getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  bool Changed = false;
  SmallVector<MachineInstr *, 8> Worklist;

  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (getGlobalSAddrOpcode(MI.getOpcode()) != ~0u)
        Worklist.push_back(&MI);

  for (MachineInstr *Load : Worklist) {
    // Non-SADDR GLOBAL_LOAD: operand 1 is the 64-bit address.
    MachineOperand &AddrOp = Load->getOperand(1);
    if (!AddrOp.isReg() || !AddrOp.getReg().isVirtual())
      continue;
    Register AddrReg = AddrOp.getReg();
    if (!AMDGPU::VReg_64RegClass.hasSubClassEq(MRI.getRegClass(AddrReg)))
      continue;

    PhiInfo Info;
    if (!analyzeAddrPhi(AddrReg, MRI, MLI, Info))
      continue;

    LLVM_DEBUG(dbgs() << "[SIPromoteGlobalLoadSAddr] promoting: " << *Load);
    Changed |= promote(*Load, Info, MRI, TII);
  }

  return Changed;
}
