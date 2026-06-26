#!/usr/bin/env python3
"""Migrate AMDGPU lit test RUN lines to the amdgpu subarch triple scheme.

The AMDGPU backend is moving from describing a target with a generic
``amdgcn`` triple plus an ``-mcpu=<gfx>`` subtarget to encoding the ISA
version directly in the triple's sub-architecture field (see the
``AMDGPU: Introduce amdgpu triple arch`` change and ``AMDGPUUsage.rst``).

This script rewrites the command-line form found in ``RUN:`` lines:

    -mtriple=amdgcn<...> ... -mcpu=gfx900   ->   -mtriple=amdgpu9.00<...>

i.e. it folds the ``-mcpu`` subtarget into the triple's subarch and drops
the now-redundant ``-mcpu`` flag. The ``-mtriple`` and ``-mcpu`` tokens may
appear in any order and need not be adjacent.

Conservative by design -- the fold above only happens when a RUN line
contains *exactly one* ``-mtriple=amdgcn*`` token and *exactly one*
``-mcpu=<cpu>`` token whose CPU is a plain gfx target (or a known device
alias of one). Lines that:

  * specify a feature-suffixed CPU (e.g. ``-mcpu=gfx900:xnack+``),
  * use a generic target (e.g. ``-mcpu=gfx9-generic``),
  * carry multiple ``-mtriple`` or ``-mcpu`` tokens, or
  * set the target via the IR (``target triple`` / ``target-cpu`` attr),

are not folded.

Additionally, for RUN lines that drive ``opt`` (and not ``llc``), the bare
arch is renamed ``amdgcn`` -> ``amdgpu`` even when there is no ``-mcpu`` to
fold, since ``amdgcn`` is now an alias of the preferred ``amdgpu`` arch and
``opt`` does not emit the triple string into its output. This bare rename is
*not* applied to ``llc`` lines, whose output embeds the triple. A few suites
(see ``BARE_ARCH_BY_PATH``) instead pin the bare arch to an explicit subarch
triple -- e.g. ``Analysis/CostModel/AMDGPU`` uses ``amdgpu6.01``, which
matches the cost model's default (no ``-mcpu``) output.

This only changes how the target is spelled on the command line; codegen is
byte-identical except where the triple string itself is emitted into output
(e.g. the ``.amdgcn_target`` directive and ``amdhsa.target`` metadata), so a
handful of tests that CHECK those strings will need their expectations
regenerated separately.

Usage:
    amdgpu-migrate-test-triples.py [--dry-run] FILE [FILE ...]

Prints the path of each file it changed (or would change, with --dry-run).
Exit status is 0 if any file changed, 1 otherwise.

The CPU->subarch tables below are derived from
llvm/include/llvm/TargetParser/AMDGPUTargetParser.def. Update them here if
new GPUs are added there.
"""
import re
import sys

# Plain gfx CPUs (AMDGCN_GPU entries in AMDGPUTargetParser.def, excluding the
# *-generic targets and the device-name aliases).
GFX = (
    "gfx600 gfx601 gfx602 "
    "gfx700 gfx701 gfx702 gfx703 gfx704 gfx705 "
    "gfx801 gfx802 gfx803 gfx805 gfx810 "
    "gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx942 gfx950 "
    "gfx1010 gfx1011 gfx1012 gfx1013 "
    "gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 "
    "gfx1100 gfx1101 gfx1102 gfx1103 "
    "gfx1150 gfx1151 gfx1152 gfx1153 gfx1154 "
    "gfx1170 gfx1171 gfx1172 "
    "gfx1200 gfx1201 gfx1250 gfx1251 gfx1310"
).split()

# Device-name aliases (AMDGCN_GPU_ALIAS entries) -> canonical gfx name.
ALIASES = {
    "tahiti": "gfx600",
    "pitcairn": "gfx601", "verde": "gfx601",
    "hainan": "gfx602", "oland": "gfx602",
    "kaveri": "gfx700",
    "hawaii": "gfx701",
    "kabini": "gfx703", "mullins": "gfx703",
    "bonaire": "gfx704",
    "carrizo": "gfx801",
    "iceland": "gfx802", "tonga": "gfx802",
    "fiji": "gfx803", "polaris10": "gfx803", "polaris11": "gfx803",
    "tongapro": "gfx805",
    "stoney": "gfx810",
}


def subarch(cpu):
    """gfx900 -> amdgpu9.00 ; gfx1250 -> amdgpu12.50 ; gfx90a -> amdgpu9.0a"""
    n = cpu[3:]  # strip 'gfx'
    return "amdgpu%s.%s" % (n[:-2], n[-2:])


# Generic targets map to "major subarch" triples (no minor version).
GENERICS = {
    "gfx9-generic": "amdgpu9",
    "gfx9-4-generic": "amdgpu9.4",
    "gfx10-1-generic": "amdgpu10.1",
    "gfx10-3-generic": "amdgpu10.3",
    "gfx11-generic": "amdgpu11",
    "gfx12-generic": "amdgpu12",
    "gfx12-5-generic": "amdgpu12.5",
}

MAP = {cpu: subarch(cpu) for cpu in GFX}
for alias, canon in ALIASES.items():
    MAP[alias] = MAP[canon]
MAP.update(GENERICS)

# -mcpu=<cpu> (also --mcpu=) where <cpu> has no trailing ':feature' suffix.
# The cpu may contain hyphens (e.g. the gfx9-4-generic targets).
MCPU_RE = re.compile(r'(?<!-)--?mcpu=([A-Za-z0-9][A-Za-z0-9-]*)(?![\w:.])')
# amdgcn arch token in a triple flag. Matches the llc/opt -mtriple= and
# --mtriple= forms as well as the llvm-objdump --triple= form.
MTRIPLE_RE = re.compile(r'(?<!-)(--?m?triple=)amdgcn(?=[-\s]|$)')


# A RUN line drives opt (and not llc) -- safe to rename the bare arch even
# without an -mcpu to fold, because opt does not emit the triple string the way
# llc does (.amdgcn_target / amdhsa.target).
def is_opt_only_line(line):
    return re.search(r'\bopt\b', line) is not None and \
        re.search(r'\bllc\b', line) is None


def rewrite_run_line(line, bare_arch="amdgpu"):
    mcpus = MCPU_RE.findall(line)
    mtriples = MTRIPLE_RE.findall(line)
    # Fold amdgcn triple(s) + -mcpu(s) into the subarch triple. A RUN line may
    # pipe several tools (e.g. opt | llc, or llc | llvm-objdump), each carrying
    # its own matching -mtriple/--triple and -mcpu. Fold them when there is one
    # amdgcn triple per -mcpu and every -mcpu names the same foldable target.
    if (mcpus and mtriples and len(mcpus) == len(mtriples)
            and len(set(mcpus)) == 1 and mcpus[0] in MAP):
        cpu = mcpus[0]
        line = MTRIPLE_RE.sub(r'\g<1>' + MAP[cpu], line)
        # Drop every -mcpu/--mcpu=<cpu> token, absorbing one adjacent space.
        mcpu_tok = r'(?<!-)--?mcpu=' + re.escape(cpu) + r'(?![\w:.-])'
        line = re.sub(r'\s' + mcpu_tok, '', line)
        line = re.sub(mcpu_tok + r'\s?', '', line)
        return line
    # Otherwise, for opt-only lines, rename the bare amdgcn arch without
    # touching any -mcpu. The replacement is normally the bare "amdgpu" alias,
    # but some suites pin the default subtarget explicitly (see process()).
    if is_opt_only_line(line):
        line = MTRIPLE_RE.sub(r'\g<1>' + bare_arch, line)
    return line


# Suites where a CPU-less opt RUN line should have its bare amdgcn arch
# replaced with an explicit subarch triple rather than the bare "amdgpu" alias.
# The cost model's default (no -mcpu) subtarget output matches amdgpu6.01.
BARE_ARCH_BY_PATH = (
    ("/Analysis/CostModel/AMDGPU/", "amdgpu6.01"),
)


def bare_arch_for(path):
    for needle, arch in BARE_ARCH_BY_PATH:
        if needle in path:
            return arch
    return "amdgpu"


def process(path, dry_run=False):
    with open(path) as f:
        text = f.read()
    bare_arch = bare_arch_for(path)
    out = []
    changed = False
    for line in text.splitlines(keepends=True):
        # Match active RUN: lines as well as disabled variants (e.g. XUN:, xUN:,
        # RUNX:) -- commented-out RUN lines kept for future re-enabling.
        is_run = re.search(r'\b(RUN|XUN|xUN|RUNX):', line) is not None
        if is_run and re.search(r'--?m?triple=amdgcn', line):
            new = rewrite_run_line(line, bare_arch)
            if new != line:
                changed = True
            out.append(new)
        else:
            out.append(line)
    if changed and not dry_run:
        with open(path, 'w') as f:
            f.write(''.join(out))
    return changed


def main(argv):
    dry_run = False
    if argv and argv[0] == '--dry-run':
        dry_run = True
        argv = argv[1:]
    any_changed = False
    for p in argv:
        if process(p, dry_run):
            print(p)
            any_changed = True
    return 0 if any_changed else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
