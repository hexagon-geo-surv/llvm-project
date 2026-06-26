#!/usr/bin/env python3
"""Pin AMDGPU lit tests that rely on the implicit default subtarget.

The AMDGPU backend now errors ("cannot codegen with no subarch") when asked
to generate code for a triple whose sub-architecture field is empty -- i.e. a
bare ``amdgcn`` triple with no subarch baked in. An ``-mcpu`` no longer lifts
the triple out of this state; only a subarch triple does. Historically a bare
``amdgcn`` triple fell back to a per-OS default subtarget, so this script makes
that previously-implicit choice explicit by rewriting the bare triple to the
matching subarch triple:

    amdhsa OS                 -> amdgpu7.00 (gfx700)
    everything else (unknown, -> amdgpu6.00 (gfx600)
    mesa3d, amdpal, bare)

Only the ``-mtriple=amdgcn[...]`` token's arch field is rewritten; the OS and
remaining triple components are preserved. The ``r600`` arch and triples that
already carry a subarch (``amdgpuN.NN``) are left alone. Any redundant
``-mcpu`` naming the same default CPU on the rewritten line is dropped.

Usage:
    amdgpu-pin-default-subtarget.py [--dry-run] FILE [FILE ...]

Prints each changed file. Exit 0 if any changed, 1 otherwise.
"""
import re
import sys

# A bare amdgcn arch token inside an -mtriple/--mtriple flag (not amdgcnN, and
# not the already-migrated amdgpu form). Matches -mtriple=amdgcn, amdgcn-,
# amdgcn--amdhsa, amdgcn| (glued to a pipe), etc.
TRIPLE_RE = re.compile(r'(--?mtriple=)amdgcn(?=[-\s|]|$)')


def subarch_for(line, m):
    """Choose the subarch triple based on the OS in the matched triple."""
    # Look at the triple's tail right after the arch to find the OS.
    tail = line[m.end():]
    os_field = re.split(r'[\s|]', tail, maxsplit=1)[0] if tail else ""
    return "amdgpu7.00" if "amdhsa" in os_field else "amdgpu6.00"


def rewrite_run_line(line):
    # Lines that name an explicit -mcpu are fold cases (the subarch should match
    # that cpu, not the historical default) -- leave them to the triple
    # migration tooling, not this default-pinning pass.
    if re.search(r'--?mcpu=', line):
        return line
    out = line
    for m in list(TRIPLE_RE.finditer(line)):
        subarch = subarch_for(line, m)
        out = TRIPLE_RE.sub(r'\g<1>' + subarch, out, count=1)
    return out


def process(path, dry_run=False):
    with open(path) as f:
        text = f.read()
    out = []
    changed = False
    for line in text.splitlines(keepends=True):
        is_run = re.search(r'\b(RUN|XUN|xUN|RUNX):', line) is not None
        if is_run and TRIPLE_RE.search(line):
            new = rewrite_run_line(line)
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
