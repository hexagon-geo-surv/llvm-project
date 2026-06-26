#!/usr/bin/env python3
"""Migrate AMDGPU lit tests that set the subtarget via a uniform IR
``"target-cpu"`` function attribute into the amdgpu subarch triple scheme.

Handles the simple case only: a single ``"target-cpu"="<gfx>"`` value that
appears in *every* attribute-group definition that the module's functions use
(so the subtarget is uniform across the module). The transform:

  1. folds the cpu into each ``-mtriple=amdgcn[...]`` token on RUN lines,
     producing the matching ``amdgpuN.NN`` subarch triple (and drops any
     redundant ``-mcpu=`` on those lines),
  2. removes the ``"target-cpu"="<gfx>"`` token from every attribute group,
  3. if removing it leaves an attribute group empty (``attributes #N = { }``),
     deletes that group definition and strips the ``#N`` reference from every
     function/define that used it.

Conservative: a file is only rewritten when
  * exactly one distinct ``"target-cpu"`` value is present and it is a known
    gfx target (or device alias),
  * that value appears only in real ``attributes #N = { ... }`` definitions
    (never inside CHECK/comment lines), and
  * there is at least one ``-mtriple=amdgcn`` RUN line to carry the subarch.

Usage:
    amdgpu-migrate-test-target-cpu.py [--dry-run] FILE [FILE ...]

Prints each changed file. Exit 0 if any changed, 1 otherwise.
"""
import re
import sys

# Reuse the CPU->subarch map from the triple migration script.
import importlib.util
import os
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "amdgpu_migrate_test_triples",
    os.path.join(_here, "amdgpu-migrate-test-triples.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MAP = _mod.MAP

TC_RE = re.compile(r'"target-cpu"="([^"]+)"')
# Allow leading indentation: MIR embeds the IR (attribute defs, defines) with
# leading whitespace inside the YAML, plain .ll has them at column 0.
ATTRDEF_RE = re.compile(r'^(\s*attributes #(\d+) = \{ )(.*)( \})\s*$')


def _is_comment_or_check(line):
    s = line.lstrip()
    return s.startswith((';', '#', '//'))


def transform(text):
    lines = text.splitlines(keepends=True)

    # Collect target-cpu values, and verify they only appear in real attr defs.
    tcs = set()
    for line in lines:
        if '"target-cpu"=' not in line:
            continue
        if _is_comment_or_check(line):
            return None  # appears in CHECK/comment -> not a clean source case
        for m in TC_RE.finditer(line):
            tcs.add(m.group(1))
    if len(tcs) != 1:
        return None
    cpu = next(iter(tcs))
    if cpu not in MAP:
        return None
    subarch = MAP[cpu]

    # The RUN lines must already, or be made to, encode this subarch. Two
    # accepted shapes:
    #  (a) a bare -mtriple=amdgcn arch we fold the cpu into, or
    #  (b) every -mtriple already the matching amdgpu<subarch> triple (the
    #      triple migration already ran); we only strip the redundant attribute.
    runs = [l for l in lines if 'RUN:' in l]
    has_bare = any(re.search(r'--?mtriple=amdgcn(?![0-9])', l) for l in runs)
    triples = set(re.findall(r'--?mtriple=(amdgpu[0-9][0-9.a-z]*)',
                             ' '.join(runs)))
    # All explicit amdgpu triples must match the attribute's subarch, and there
    # must be no other amdgcn-family triple in play.
    already_ok = (not has_bare and triples and
                  all(t == subarch or t.startswith(subarch + '-')
                      for t in triples))
    if not (has_bare or already_ok):
        return None

    # Pass 1: rewrite RUN lines (fold cpu into a bare amdgcn triple, drop any
    # redundant -mcpu). No-op when the triple is already the right subarch.
    out = []
    for line in lines:
        if 'RUN:' in line and re.search(r'--?mtriple=amdgcn(?![0-9])', line):
            line = re.sub(r'(--?mtriple=)amdgcn(?=[-\s]|$)',
                          r'\g<1>' + subarch, line)
            # Drop any -mcpu= on this RUN line (the subarch now sets the cpu).
            line = re.sub(r'\s--?mcpu=[A-Za-z0-9:+-]+', '', line)
        out.append(line)
    lines = out

    # Pass 2: remove target-cpu from attribute defs; find emptied groups.
    emptied = set()
    out = []
    for line in lines:
        md = ATTRDEF_RE.match(line)
        if md and '"target-cpu"' in md.group(3):
            body = md.group(3)
            body = re.sub(r'\s*"target-cpu"="[^"]+"', '', body).strip()
            if body == '':
                emptied.add(md.group(2))
                continue  # drop the now-empty group definition
            out.append('%s%s%s\n' % (md.group(1), body, md.group(4)))
        else:
            out.append(line)
    lines = out

    # Pass 3: strip references to emptied groups from defines/declares.
    if emptied:
        # Build a regex matching " #N" where N is an emptied group, as used in
        # function headers (define/declare ... #N ...).
        def strip_refs(line):
            s = line.lstrip()
            if not (s.startswith('define') or s.startswith('declare')):
                return line
            for n in emptied:
                line = re.sub(r' #' + n + r'\b', '', line)
            return line
        lines = [strip_refs(l) for l in lines]

    return ''.join(lines)


def process(path, dry_run=False):
    with open(path) as f:
        text = f.read()
    new = transform(text)
    if new is None or new == text:
        return False
    if not dry_run:
        with open(path, 'w') as f:
            f.write(new)
    return True


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
