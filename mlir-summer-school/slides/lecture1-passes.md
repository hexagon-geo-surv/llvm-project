---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 25px; }
  section pre { font-size: 0.6em; line-height: 1.2; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8em; }
  section.lead { text-align: center; }
---

<!-- _class: lead -->

# Your First Pass
## IR surgery with the C++ API

**MLIR Summer School — Transformations (1/3)**

**Matthias Springer**

<!-- Speaker notes:
Welcome to the Transformations module. Session budget (canonical): 0:00-0:05 warm-up quiz | 0:05-0:45 lecture incl. embedded quizzes+demos (~40 min core path) | 0:45-0:50 exercise briefing | 0:50-1:20 hands-on (30 min) | 1:20-1:30 solution walkthrough + wrap-up.
Core path ≈40 min: slides marked ⏱ are the pressure-release valves — skip them if behind schedule, in this order: LICM walk+library-call (+2), pass options & statistics (+2), InsertionGuard (+1), debugging toolkit (2) (+2), pipeline quiz + answers as a pair (+3), why-anchors-matter threading (+2), textual pipeline syntax (+2), PassManager tree (+2), failing & staying honest (+2), WalkResult (+1), traversal level 0 (+1), constants/Location (+1), nesting gotcha (+1), navigation map (+1), getDefiningOp-crash quiz + answers as a pair (+3), wrong-insertion-point quiz + answers as a pair (+3).
Prerequisites the audience already has: compiler basics (SSA, "what is a pass"), MLIR IR structure (ops/regions/blocks/values), ODS/TableGen. They have NOT written C++ against the MLIR API yet — today is that day.
Setup check (do it BEFORE 0:00): everyone should have the prebuilt LLVM/MLIR tree and the exercises/ project configured. All demos in this deck run with build/bin/mlir-opt from the llvm-project checkout root.
~30 s.
-->

---

# Today's route

1. **Warm-up quiz** — IR anatomy in your own vocabulary
2. **The C++ object model** — `Operation*`, `Region`, `Block`, `Value`
3. **Generic vs. generated ops** — `isa` / `dyn_cast` / `cast`
4. **Use-def chains** — follow the wires, then rewire them
5. **`walk()`** — traversing IR without writing loops
6. **`OpBuilder`** — creating new ops at the right place
7. **Worked example** — strength-reduce `muli` → `shli`
8. **Pass anatomy** — TableGen def → generated base → `runOnOperation`
9. **PassManager & pipelines** — nesting, threading, the CLI
10. **Debugging toolkit** — the flags you'll use every day
11. **Exercise 1** — you build the pass yourself

<!-- Speaker notes:
Quick orientation, don't dwell. Points 2-7 are "the IR mutation API", points 8-10 are "the pass infrastructure around it". The exercise combines both.
~30 s. Leaving this slide you should be ~2 min into the session; the warm-up quiz + answers fill the rest of the 0:00-0:05 block.
-->

---

# 🧠 Quiz: warm-up — point at the pieces

```mlir
func.func @warmup(%a: i32, %n: index) -> i32 {
  %c2 = arith.constant 2 : i32
  %r = arith.muli %a, %c2 {answer = 42 : i64} : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %r) -> (i32) {
    %next = arith.addi %acc, %a : i32
    scf.yield %next : i32
  }
  return %sum : i32
}
```

Call out **every single one** of: **① operations ② values ③ block arguments ④ attributes ⑤ regions**

<!-- Speaker notes:
Show of hands / call on people, one item at a time — push for completeness ("keep going, what else?"), not just one example per category. This recaps their IR-structure session in their own words. The snippet round-trips through mlir-opt unchanged (verified).
Answers (no separate answer slide — deliver them verbally right here):
① Operations: func.func, arith.constant (x3), arith.muli, scf.for, arith.addi, scf.yield, func.return. (9 as written; 10 counting the implicit builtin.module that mlir-opt wraps around it — worth mentioning even though it's not asked.)
② Values: %a, %n, %c2, %r, %c0, %c1, %sum, %i, %acc, %next — every %-thing.
③ Block arguments: %a, %n (entry block of the func body region), and %i, %acc (scf.for body block). The loop IV being a block argument surprises people — good discussion moment.
④ Attributes: {answer = 42 : i64} on the muli; also the 2 : i32 payload of arith.constant, the function name @warmup (sym_name), and the function type — attributes are everywhere, not just in braces.
⑤ Regions: the func body, and the scf.for body.
Key takeaways to fold in while giving the answers: (1) attributes are compile-time constant data attached to an op — not just the {…} dictionary, also constant payloads and function names; (2) the loop IV is a block argument — MLIR has no phi nodes, block arguments play that role; (3) values come in exactly two forms — that's literally the C++ class design, coming up next.
~2 min — keep it rapid-fire (one item at a time); the warm-up block ends at 0:05.
-->

---

# From words to classes: the C++ names

| In the text… | …is this C++ class |
|---|---|
| an **operation** (`arith.muli`, `scf.for`, …) | `Operation` |
| a **value** (any `%name`) | `Value` → `OpResult` / `BlockArgument` |
| a **use** of a value/block | `OpOperand` / `BlockOperand` |
| an **attribute** (`{answer = 42 : i64}`, …) | `Attribute` (e.g. `IntegerAttr`, `StringAttr`, …) |
| a **region** (the `{ ... }` body) | `Region`, containing `Block`s |

<!-- Speaker notes:
Bridge slide: every category from the warm-up quiz gets its exact C++ name here, before the object-model slides start using them without re-introducing them.
The split of "value" into OpResult/BlockArgument previews the "Value: exactly two forms" slide later — don't over-explain here, just plant the names.
"Use" is subtle and easy to skip past: a value/block doesn't hold its uses, its USERS do — each operand slot or branch target IS an OpOperand/BlockOperand. This previews "Use-def chains" (the edge that knows both endpoints) and the later OpOperand/BlockOperand deep-dive slide — again, just plant the two names here, don't explain the mechanism yet.
Attribute is deliberately shown as an umbrella too, mirroring Value — {answer = 42 : i64} is concretely an IntegerAttr; a string attribute would be StringAttr; etc. Same casting story (isa/dyn_cast) applies to Attribute as to Operation and Value — foreshadow only, full treatment isn't in this lecture.
Block has no dedicated bullet in the quiz (only "block ARGUMENT" does) — call that out explicitly, it's a deliberate gap the quiz left open, and this slide closes it.
~1 min.
-->

---

# The big picture: how the core IR classes relate (simplified)

```text
                          ┌──────────────────────────────────────┐
                     ┌──┐ │   ┌──────────────────────────────┐   │  ┌──┐
                     │  ▼ ▼*  │*                             ▼   │  │  ▼
 ┌────────────┐ *   ┌───────────┐      * ┌────────┐      * ┌───────────┐      * ┌───────────────┐
 │ ConcreteOp │ ──► │ Operation │ ◆───── │ Region │ ◆────▶ │   Block   │ ◆────▶ │ BlockArgument │
 └────────────┘     └───────────┘ ◀───── └────────┘ ◀───── └───────────┘ ◀───── └───────────────┘
                         ◆                                      ▲  │                  │
                         │                                      │  │                  │
                         │                                      │  │                  │
                         ├──────────────────┬────────────────┐  │  │                  │
                         │*                 │*               │* │* ▼                  │
                    ┌───────────┐      ┌──────────┐     ┌──────────────┐              │
                    │ OpOperand │      │ OpResult │     │ BlockOperand │              │
                    └───────────┘      └──────────┘     └──────────────┘              │
                      │  ▲  │* ▲            │                │   ▲                    │
                      └──┘  │  │            ▽                └───┘                    │
                            │  │       ┌─────────┐                                    │
                            │  └────── │  Value  │ ◁──────────────────────────────────┘
                            └────────► └─────────┘
```

`─▶` Pointer     `◆─` Aggregate (coupled lifetime)     `─▷` Inheritance

<!-- Speaker notes:
FLEX SLIDE — capstone/overview, present after (or in place of) the individual deep-dive slides if short on time; if skipped, say: everything's embedded until the graph recurses into a new Block or Operation — that's the one-sentence summary of the whole memory-layout arc.
The loop above Operation and above Block is the same Prev/Next mechanism drawn twice, once per container: Operation is spliced into its parent Block's operation-list, Block is spliced into its parent Region's block-list.
All three children hanging off Operation (`OpOperand`, `Value`, `BlockOperand`) are aggregated the same way — that's why all three get their own `◆`, not just Region. `OpOperand ──► Value` and `BlockOperand ──► Block` are the two "outgoing" reference edges from those aggregated children back into the rest of the graph.
Simplified deliberately — relationships NOT drawn here, mention only if asked: Operation's own `block` field (a reference up to its PARENT Block, separate from the Prev/Next loop); Block's `arguments` vector (references Value/BlockArgumentImpl, not embedded); Block's `firstUse` (references a BlockOperand living inside some OTHER op); Value's `firstUse` (references the head of its OpOperand use-chain); OpOperand/BlockOperand also have their own Prev/Next-style loops (nextUse/back) for their use-chains, omitted here to avoid a third loop shape.
The recursion loop (Block ──► Operation) is THE loop that makes IR a tree of arbitrary depth — nested regions inside nested regions inside nested regions, all through this one edge repeating.
~2 min.
-->

---

# The C++ object model: it's `Operation*` all the way down

At runtime, *every* op — `arith.addi`, `scf.for`, `func.func`, even `builtin.module` — is a `mlir::Operation`. Take one concrete, fully-written-out example:

```mlir
%r = arith.muli %a, %b : i32                    // custom assembly format (pretty syntax)
%r = "arith.muli"(%a, %b) : (i32, i32) -> i32   // generic syntax — every op supports this
```

```text
Operation "arith.muli"           (always heap-allocated; handled as Operation*)
 ├─ name        "arith.muli"
 ├─ location    Location                  ← file:line, or unknown
 ├─ operands    [OpOperand, OpOperand]     ← exactly 2: edges to %a, %b (defined elsewhere)
 ├─ results     [Value : i32]              ← exactly 1: %r, the value THIS op defines
 ├─ successors  []                         ← empty — muli isn't a branch, has no successors
 ├─ attributes  {}                         ← empty — no *discardable* attributes here
 ├─ properties  { overflowFlags: none }    ← the ODS "overflowFlags" arg — default, elided when printed
 └─ regions     []                         ← empty — muli has no nested regions
```

The textual IR you read **is** a 1:1 serialization of this in-memory object graph — there's no separate AST.

<!-- Speaker notes:
This is one of the two most important slides of the first half (the other is the regions version, two slides from now). There is no separate AST: what you print is what's in memory.
The generic syntax line maps almost literally onto the diagram's rows: the quoted `"arith.muli"` is the name field, `(%a, %b)` are the operands, `(i32, i32) -> i32` are the operand/result types — every op, registered or not, prints this way; `arith.muli %a, %b : i32` is just a nicer custom form the dialect opted into. Good one-liner if asked "how does mlir-opt know how to print this without a custom printer?": it doesn't — it falls back to generic syntax.
Emphasize: Operation is uniform — a module and an addi are the same C++ class, differing only in name, operand/result/successor counts, attributes, properties, and regions. This uniformity is what makes generic passes possible.
Deliberately a boring, fully-concrete example (2 operands, 1 result, no successors, no discardable attrs, no regions) so every row of the diagram maps to something visible in the one line of IR above it — no hidden defaults to explain away yet, except properties (see below).
"operands" are OpOperand, not raw Value — an OpOperand is a Value edge that also remembers its owner (needed later for use-lists; the "Use-def chains" slide names it properly and explains why). "results" are handed back to you as Value/OpResult, but Operation does not literally store a field of OpResult objects — the later flex slide ("what's really in memory") shows the real, more surprising storage.
attributes vs. properties, the subtlety: {answer = 42} on the warm-up slide's muli is a discardable ATTRIBUTE (arbitrary, in the generic attrs dictionary); overflowFlags is an ODS-declared inherent argument stored as a PROPERTY (compact POD data, not in attrs) — that's why attributes can be truly {} here while the op still carries ODS-level data. Full mechanics on the "what's really in memory" flex slide (propertiesStorageSize, OpProperties row).
successors only matter for branch-like ops (cf.br, cf.cond_br); muli has ::mlir::OpTrait::ZeroSuccessors — verified in the generated MulIOp class (ArithOps.h.inc).
Handle discipline: Operation is always passed as Operation* (pointer) or Operation& — never by value.
~2 min.
-->

---

# Building that op, the fully generic way

No ODS, no typed op class — just `OperationState`, the same mechanism the *parser* uses for every op in a `.mlir` file (and the only way to build an **unregistered** op):

```cpp
Value a = ..., b = ...;                     // %a, %b already exist as Values
Location loc = ...;

OperationState state(loc, "arith.muli");
state.addOperands({a, b});                  // exactly 2 operands
state.addTypes(b.getType());                // exactly 1 result, type i32
// no addAttribute(...)
// no addRegion()

Operation *op = Operation::create(state);   // create a "detached" operation (no owner)
op->dump();                                 // dump to stderr
op->destroy();                              // deallocate memory (incl. nested IR)
```

`OperationState` is a plain builder-of-a-struct: you fill in name, operands, result types, attributes, successors, regions — by hand — then `Operation::create` allocates the real `Operation` from it.

<!-- Speaker notes:
Intentionally the "hard way" — nobody writes this in day-to-day pass code, but seeing it once demystifies what OpTy::build/create actually do (next slide): they are convenience wrappers that fill in exactly this struct for you, using the ODS argument names you already know.
Mention in passing: this is also how mlir-opt's textual parser builds ops, and the only path available for ops from dialects the tool doesn't have compiled in (-allow-unregistered-dialect).
Deliberately not talking about insertion/OpBuilder yet — that's its own section later ("OpBuilder: a cursor into the IR" onward). For now this op simply exists, fully formed, wherever it happens to sit.
~2 min.
-->

---

# Building that op with the generated `build()` function

Same two steps as the slide before — `OperationState`, then `Operation::create` — but the *filling-in* step is now generated instead of hand-written:

```cpp
Value a = ..., b = ...;               // %a, %b already exist as Values
Location loc = ...;

OperationState state(loc, arith::MulIOp::getOperationName());
arith::MulIOp::build(builder, state, a, b, /*overflowFlags=*/{});  // fills in operands/types/attrs
Operation *op = Operation::create(state);                          // create a "detached" operation
op->dump();                                                        // dump to stderr
op->destroy();                                                     // deallocate memory (incl. nested IR)
```

The `build()` function is auto-generated from the ODS definition.

<!-- Speaker notes:
The bridge slide: same skeleton as "the fully generic way", one step swapped from hand-written to generated.
build() overloads mirror the build(...) forms students already saw in the ODS session — one for each way of constructing the op (with/without explicit result types, with/without optional attributes, etc.).
The `builder` parameter here is required by the generated signature but not doing anything insertion-related in this step — that's deliberately not the topic yet; `OpTy::create(builder, loc, ...)` (later, once OpBuilder is introduced properly) is where build() and insertion actually get fused into one call.
~2 min.
-->

---

# The object model, with regions: `scf.for`

An op with nested IR (regions and blocks).

```mlir
%sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %r) -> (i32) {
  %next = arith.addi %acc, %a : i32
  scf.yield %next : i32
}
```

```text
Operation "scf.for"                       (heap-allocated)
 ├─ operands    [OpOperand x4]             %c0, %n, %c1, %r
 ├─ results     [Value : i32]              %sum
 └─ regions     [Region]                   embedded IN this op's allocation — not separate!
        Region
         └─ blocks  [Block*]               Block IS separately heap-allocated (`new Block`)
              Block
               ├─ arguments  [BlockArgument x2]        %i, %acc
               └─ operations [Operation*, Operation*]  heap-allocated ops — recursion!
                    Operation "arith.addi"
                    Operation "scf.yield"
```

<!-- Speaker notes:
Directly answers "aren't Blocks also heap-allocated?": yes, unambiguously — `new Block` is a real, separate allocation, unlike Region.
The nesting you know from the textual IR IS the C++ ownership structure: ops own regions (embedded), regions own blocks (a linked list of separately-allocated Blocks), blocks own ops (recursion) — there is no separate AST.
Operand count reminder: lowerBound, upperBound, step, then one operand per iter_arg (%r here) — 4 total for one iter_arg; results mirror the iter_args 1:1 (1 result here, %sum).
This is the slide to point back to during "Traversal, level 0" — that recursive printOperation function walks exactly this structure.
~2 min.
-->

---

# ⏱ `Operation`s in a `Block` form a doubly-linked list

```text
┌──────────────────────────────────────────┐
│ ilist_node_with_parent<Operation, Block> │
├──────────────────────────────────────────┤
│ NodeBase *Prev, *Next                    │
└─────────────────────△────────────────────┘
                      │
                ┌───────────┐
                │ Operation │
                └───────────┘
```

Example:
```
^bb0:
"test.dummy_op1"() : () -> ()
"test.dummy_op2"() : () -> ()
"test.dummy_op3"() : () -> ()
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: Operation inherits ilist_node_with_parent<Operation, Block> for its Prev/Next splice-into-Block.operations linkage — same trick as Block splicing into Region.blocks — and separately (privately) TrailingObjects, which contributes no fields, just the offset arithmetic for the layout on the next slide.
Prev/Next here are NOT the same field as Operation's own `block` pointer (next slide) — `block` is the PARENT pointer (which Block owns me), Prev/Next are the SIBLING links (who comes before/after me in that Block's operation list). Two different relationships, two different sources: one is Operation's own field, the other is inherited.
~1 min.
-->

---

# ⏱ What's *really* in memory: `class Operation`'s fields

One `malloc`, three zones — results are a **prefix**, almost everything else is in the **suffix**:

```text
               ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
               ├─────────── PREFIX: results (one OpResultImpl per result, in reverse order) ──────────────────────┤
               │ OpResultImpl[numResults]                // OpResultImpl = { Type, use-list head }                │
Operation* ──► ├───────────────────────────────── the `Operation` object itself ──────────────────────────────────┤
               │ Block *block;                                                                                    │
               │ Location location;                                                                               │
               │ unsigned orderIndex;                    // cached position within block (mutable)                │
               │ const unsigned numResults, numSuccs;                                                             │
               │ unsigned numRegions : 23;                                                                        │
               │ bool hasOperandStorage : 1;                                                                      │
               │ unsigned propertiesStorageSize : 8;                                                              │
               │ OperationName name;                                                                              │
               │ DictionaryAttr attrs;                                                                            │
               ├─────────────── trailing, right after the object (order per TrailingObjects<...>) ────────────────┤
               │ OperandStorage             // { capacity:31, isStorageDynamic:1, numOperands, OpOperand *ptr }   │
               │ OpProperties               // propertiesStorageSize bytes of ODS-declared POD data               │
               │ BlockOperand[numSuccs]     // inline array of successors                                         │
               │ Region[numRegions]         // inline array of regions                                            │
               │ OpOperand[numOperands]     // inline array of operands                                           │
               └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: Operation is one allocation sized exactly for its result/operand/region/successor counts, with results stored as a prefix and operands/regions/successors trailing — the next slide's takeaway (what's separate vs. embedded) is the one to deliver verbally if this is skipped.
The arrow is verified straight from the header's own doc comment (Operation.h:41-47): "[Result2, Result1, Result0, Operation] ^ this is where `Operation*` pointer points to." Point at it explicitly — the pointer everyone holds points at the START of the object's own fields, never at the prefix.
The surprising bit worth calling out explicitly: OpResult is NOT a stored field — what's stored (as a prefix, before the Operation pointer) is an array of OpResultImpl (InlineOpResult for the first few results, OutOfLineOpResult beyond a small inline capacity), each just {Type, use-list head}. Operation::getResult(i) constructs an OpResult (owner + result number) on demand from that storage.
Why a prefix AND trailing, instead of one side? Historical/perf: it lets Operation* stay a stable "middle" pointer that both directions can offset from with O(1) index arithmetic, and keeps the common no-result/no-operand case allocation-free on that side.
OperandStorage vs. OpOperand[]: don't let students conflate the two rows — OperandStorage is a small, ALWAYS-present header (capacity/count/pointer); the OpOperand array it points at is usually (not always!) the trailing slot shown right below. This is the one place in the whole layout where "it's all one allocation" can be false for a *specific* op instance, not just categorically (like Block/Region) — worth calling out if anyone's mental model was "everything trailing is permanently fixed at construction".
attrs is a single DictionaryAttr — MLIR does not allocate one heap object per attribute; the whole {k: v, ...} dictionary is one shared, immutable object, and {} is a shared empty instance (no allocation on the fast path).
Don't derive a permanent mental model from the exact TrailingObjects order — treat this as "here's evidence it's compact and correct", not something you need memorized to write passes.
The Region row (embedded, not a pointer!) is the hook into the next slide.
(+2 if presented.)
-->

---

# 🧠 Quiz (1/6): change a result's type

Starting op: `%r = "test.op"(%a, %b) : (i32, i32) -> i32`

Target: `%r = "test.op"(%a, %b) : (i32, i32) -> i64`

How would you get there?

<!-- Speaker notes:
Give ~30 seconds per case; this is the payoff quiz for the whole memory-layout arc, and every answer traces back to a field from the box two slides ago. Don't reveal the in-place/new-op split via the question wording; let them reason it out from the fields.
~30 s.
-->

---

# ✅ (1/6): change a result's type

```cpp
op->getResult(0).setType(b.getI64Type());
```

**In place.** A result's type lives in `OpResult` — a single-field overwrite.

<!-- Speaker notes:
~30 s.
-->

---

# 🧠 Quiz (2/6): add a result

Starting op: `%r = "test.op"(%a, %b) : (i32, i32) -> i32`

Target: `%r, %s = "test.op"(%a, %b) : (i32, i32) -> (i32, i32)`

How would you get there?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ (2/6): add a result

**Can't be done in place** — `numResults` is `const`; results are a fixed-size prefix.

Must build a new op with both result types, then:

```cpp
op->getResult(0).replaceAllUsesWith(newOp->getResult(0));  // only %r had prior uses
op->erase();
```

<!-- Speaker notes:
Note the asymmetry vs. a same-arity replacement: old op has 1 result, new op has 2 — you RAUW only the corresponding result (index 0 → index 0), not the whole ResultRange (mismatched counts would assert). %s is brand new; nothing to replace it with.
~30 s.
-->

---

# 🧠 Quiz (3/6): change an operand

Starting op: `%r = "test.op"(%a, %b) : (i32, i32) -> i32`

Target: `%r = "test.op"(%c, %b) : (i32, i32) -> i32`

How would you get there?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ (3/6): change an operand

```cpp
op->setOperand(0, c);
// alternative: get the OpOperand first, then assign through it
// op->getOpOperand(0).set(c);
```

**In place.** Overwrite the `OpOperand`'s `value` field (and update linked lists).

<!-- Speaker notes:
~30 s.
-->

---

# 🧠 Quiz (4/6): add an operand

Starting op: `%r = "test.op"(%a, %b) : (i32, i32) -> i32`

Target: `%r = "test.op"(%a, %b, %c) : (i32, i32, i32) -> i32`

How would you get there?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ (4/6): add an operand

```cpp
op->insertOperands(2, {c});
```

**In place.** `OperandStorage` is the one growable part — it can `malloc` a bigger buffer, transparently.

<!-- Speaker notes:
Aside if asked (not the point here): this succeeds at the C++ level on any op, but a fixed-arity op like arith.muli would fail verification afterward — "compiles fine, verifier catches it," same as earlier in the deck. Not relevant for this generic test.op example.
~30 s.
-->

---

# 🧠 Quiz (5/6): add an attribute

Starting op: `%r = "test.op"(%a, %b) : (i32, i32) -> i32`

Target: `%r = "test.op"(%a, %b) {answer = 42 : i64} : (i32, i32) -> i32`

How would you get there?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ (5/6): add an attribute

```cpp
op->setAttr("answer", b.getI64IntegerAttr(42));
```

**In place.** `attrs` is one `DictionaryAttr` field; setting an attribute just swaps it for a new dictionary.

`DictionaryAttr`'s own definition (`DictionaryAttrStorage`) is just as thin — an array of `NamedAttribute`, itself just a `{name, value}` pair of `Attribute`s (**not** separately uniqued):

```cpp
struct DictionaryAttrStorage : public AttributeStorage {
  ArrayRef<NamedAttribute> value;   // NamedAttribute = { Attribute name; Attribute value; }
};
// for this example: NamedAttribute{ StringAttr("answer"), IntegerAttr(42 : i64) }
```

⚠️ **All attributes are singletons, uniqued in the `MLIRContext`.** Creating `{answer = 42 : i64}` (or any attribute) allocates it in the context's arena (`BumpPtrAllocator`) — and it stays there for the **rest of the context's lifetime**, even if nothing references it anymore. No refcounting, no GC: creating many distinct attributes in a long-running context is memory that never comes back until the context itself is destroyed.

<!-- Speaker notes:
C++ API details (setAttr, insertOperands, etc.) are previewed here but formally introduced later in the OpBuilder/rewriter sections — for now the point is just "can the existing Operation do this in place?".
DictionaryAttrStorage verified at build/tools/mlir/include/mlir/IR/BuiltinAttributes.cpp.inc:254-277 — exactly one field, an ArrayRef<NamedAttribute>, copied into the allocator via allocator.copyInto(value). NamedAttribute (Attributes.h:164-212) is NOT uniqued — it's a plain 2-pointer value type {Attribute name, Attribute value}, built transiently and copied by value into the DictionaryAttr's own array; only the array AS A WHOLE goes through the uniquer, as one DictionaryAttr.
If asked "how many new uniqued objects does this one setAttr call create (cold cache)?": 4, not 3 — b.getI64IntegerAttr(42) uniques an i64 IntegerType AND an IntegerAttr(42) (2 objects, easy to undercount as 1); setAttr(StringRef,...) uniques a StringAttr("answer") for the name (easy to forget entirely); then setAttr(StringAttr,...) builds a scratch NamedAttrList (ordinary heap memory, NOT uniqued) and calls attributes.getDictionary(ctx), which uniques exactly one new DictionaryAttr. Total: 1 Type + 3 Attributes = 4. NamedAttribute/NamedAttrList are never separately interned — swap them out of any "what gets interned" count.
The lifetime warning generalizes to ALL attributes/types, not just DictionaryAttr — StorageUniquer::StorageAllocator (mlir/include/mlir/Support/StorageUniquer.h:93-137) wraps a plain llvm::BumpPtrAllocator: bulk-freed only when the MLIRContext dies, never per-object. Practical consequence: passes that mint a fresh, never-reused attribute per op/iteration (e.g. unique debug tags) leak for the process's lifetime — prefer reusing/interning your own attribute values where possible.
~45 s.
-->

---

# 🧠 Quiz (6/6): add a region

Starting op: `%r = "test.op"(%a, %b) : (i32, i32) -> i32`

Target: `%r = "test.op"(%a, %b) ({ ^bb0: }) : (i32, i32) -> i32`

How would you get there?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ (6/6): add a region

**Can't be done in place** — `numRegions` is `const` too; regions are a fixed-size suffix.

Must build a new op with the region attached, then RAUW + erase the old one — same recipe as 2/6.

<!-- Speaker notes:
The pattern to land across all six: COUNTS baked into the layout at construction (numResults, numRegions, numSuccs) are permanently fixed — change them by building a new op and replacing the old one. CONTENT within an existing slot (a result's type, an operand's value, the attrs dictionary) is freely mutable — that's just overwriting a field. Operand COUNT is the one exception that's also mutable, via the OperandStorage indirection, because rewriting operand lists is so common in passes.
~30 s.
-->

---

# ⏱ Zooming into `arith::AddIOp`: the typed lens

```text
┌────────────────────────────────────────────────────────────────┐
│                            OpState                             │
├────────────────────────────────────────────────────────────────┤
│ Operation *state                                               │
└────────────────────────────────△───────────────────────────────┘
                                 │
┌────────────────────────────────────────────────────────────────┐
│ Op<AddIOp, ZeroRegions, OneResult, NOperands<2>, ...20 traits> │
└────────────────────────────────△───────────────────────────────┘
                                 │
                            ┌────────┐
                            │ AddIOp │
                            └────────┘
```

- `getLhs()` → `Value`, literally `getOperation()->getOperand(0)`
- `getLhsMutable()` → `OpOperand&`, literally `getOperation()->getOpOperand(0)`
- `getResult()` → `Value`, literally `getOperation()->getResult(0)`
- `->` is overloaded too: `addOp->erase()` means `addOp.getOperation()->erase()`

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: AddIOp is Op<AddIOp, ~20 trait mixins> is OpState — the only real field in the whole chain is OpState's Operation *state; every trait mixin (ZeroRegions, OneResult, NOperands<2>, IsCommutative, ...) is an empty CRTP base adding methods only, never fields. sizeof(AddIOp) == sizeof(Operation*).
This is the payoff for the "Two views of one op: generic vs. generated" slide from earlier — now with the real generated code instead of a description. getLhs() goes through getODSOperands(0), which is just std::next(getOperation()->operand_begin(), 0) — i.e. op->getOperand(0) with the index baked in at compile time from the ODS argument's position. No name lookup at runtime, ever.
getLhsMutable() returns an OpOperand&, not a Value — direct callback to the OpOperand deep-dive slide: this is literally the same edge type, letting you rewire just this one operand slot without touching the others.
Simplification for the slide: the real generated code wraps the return of getLhs()/getResult() in cast<TypedValue<Type>>(...) — a thin Value subclass asserting a known type. Saying "returns a Value" is accurate enough here; mention TypedValue only if asked.
~2 min.
-->

---

# ⏱ Zooming into `Region`: it's almost nothing

`Region`'s entire field list — this is why "embedded, not a pointer" is so cheap:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                                    Region                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ BlockListType blocks;   -- doubly-linked: head/tail ptr to first/last Block │
│ Operation *container;   -- back-pointer to the owning Operation             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Two fields, zero allocation of its own. Only the `Block`s that `blocks` points at are separately heap-allocated (`new Block`) — the recursion starts there, on the next slide.


<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: Region is just two fields (a Block list header + a back-pointer) embedded inside its owning Operation — no allocation of its own; Block is where the "separate allocation" starts, and the next slide recurses from there.
Precision for Q&A: BlockListType is llvm::iplist<Block>, which technically holds ONE embedded Sentinel node with Prev/Next pointers (not two separate head/tail fields) — but "acts like a head+tail pointer pair" is the right intuition and all that's needed here. The same trick appears one level down for Block's own op list (next slide).
Intrusive vs. std::list, one sentence if asked: std::list<Block> would allocate a wrapper node per element; iplist<Block> instead makes Block itself carry Prev/Next (via ilist_node_with_parent) — no wrapper, no extra allocation, that's the whole point of "intrusive".
container is only used going "up" (getParentOp()); walking "down" always goes through blocks — front()/back()/getBlocks().
This is a good moment to physically point back at the scf.for slide's diagram and trace: Operation (1 alloc) → Region (embedded, free) → Block (new Block, 1 alloc per block) → Operation (recursion, 1 alloc per nested op).
(+1 if presented.)
-->

---

# ⏱ Zooming into `Block`

`Block` has two base classes (multiple inheritance) plus its own fields:

```text
┌───────────────────────────────────┐    ┌───────────────────────────────────────┐
│ IRObjectWithUseList<BlockOperand> │    │ ilist_node_with_parent<Block, Region> │
├───────────────────────────────────┤    ├───────────────────────────────────────┤
│ IROperandBase *firstUse           │    │ NodeBase *Prev, *Next                 │
└─────────────────△─────────────────┘    └───────────────────△───────────────────┘
                  │                                          │
                  └────────────────────┬─────────────────────┘
                                       │
           ┌───────────────────────────────────────────────────────┐
           │                         Block                         │
           ├───────────────────────────────────────────────────────┤
           │ PointerIntPair<Region*,1,bool> parentValidOpOrderPair │
           │ OpListType operations                                 │
           │ std::vector<BlockArgument> arguments                  │
           └───────────────────────────────────────────────────────┘
```

- **`Prev`/`Next`** — inherited; a doubly-linked list
- **`firstUse`** — inherited; chains through *other* ops' successor edges (who branches to me)
- **`operations`** — same doubly-linked head/tail trick as `Region::blocks`


<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: Block is where "everything's embedded" finally breaks — operations is an embedded Sentinel just like Region::blocks, but arguments is a genuine std::vector with its own heap buffer, and each BlockArgument points at a separately new'd BlockArgumentImpl.
Precision for Q&A: each BlockArgument is a 1-pointer handle (like OpResult) to its own separately-`new`'d BlockArgumentImpl { Type, Block *owner, index, Location } — Value.h ~L280 for the struct, Value::create ~L327 for the `new detail::BlockArgumentImpl(...)` call. IRObjectWithUseList lives in UseDefLists.h.
firstUse ties back to the "Use-def chains" slide's OpOperand/use-list story — it's the SAME mechanism, just for Block-as-branch-target instead of Value-as-data. Concretely: cf.br's successor list is a BlockOperand[] trailing the cf.br Operation (see the Operation box); Block.firstUse is the head of the chain through all such edges pointing at this block.
Why does arguments get to be a plain std::vector instead of something fancier? Block arguments are rarely hot-path-critical the way operands/results are (they don't need O(1) index arithmetic from a stable Operation* the way OpResult does) — a std::vector is simplest and good enough.
Full recursion recap, all three slides: Operation (1 alloc, results prefix + operands/regions/successors trailing) → Region (embedded, free) → Block (1 alloc, operations embedded, arguments a real vector of separately-allocated BlockArgumentImpls) → Operation (recursion).
(+1 if presented.)
-->

---

# 🧠 Quiz (1/7): how fast is the i-th `OpResult`?


What's the complexity of `op->getResult(i)`?

<!-- Speaker notes:
Give ~30 seconds per case; same format as the "small edits" quiz — no separate answer slide beyond this pair. The pattern to land across all seven: arrays give O(1) random access; the intrusive doubly-linked lists don't — but those same linked lists make insert/erase O(1) once you already have a position, which plain arrays can't do. Two sides of the same tradeoff.
~30 s.
-->

---

# ✅ O(1)

`op->getResult(i)` is **O(1)**.

Results are a **PREFIX array** (`OpResultImpl[numResults]`) — direct arithmetic (inline vs. out-of-line branch on a constant threshold), never a scan.

<!-- Speaker notes:
~30 s.
-->

---

# 🧠 Quiz (2/7): how fast is the i-th `Region`?

What's the complexity of `op->getRegion(i)`?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ O(1)

`op->getRegion(i)` is **O(1)**.

Regions are a **TRAILING array** (`MutableArrayRef<Region>`) — `op->getRegions()[i]`, plain indexing.

<!-- Speaker notes:
~30 s.
-->

---

# 🧠 Quiz (3/7): how fast is inserting an op into a `Block`?

What's the complexity of inserting an operation into a block at a given location, e.g. `op->moveBefore(anotherOp)`?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ O(1)

`op->moveBefore(anotherOp)` is **O(1)**.

`Block::operations` is `llvm::iplist<Operation>`, a **doubly-linked list**. Insertion into a linked list is O(1).

<!-- Speaker notes:
Contrast with std::vector::insert, which is O(n) because of the shift. This is exactly why the IR uses an intrusive linked list instead of an array here.
~30 s.
-->

---

# 🧠 Quiz (4/7): how fast is erasing an op from a `Block`?

What's the complexity of `op->erase()`?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ O(1)

`op->erase()` is **O(1)** for ops without nested IR.

Same reasoning as insert: removing a node from a doubly-linked list is a constant number of pointer updates.

However, `op->erase()` must also erase all nested blocks, block arguments, operations.

<!-- Speaker notes:
Caveat if asked: erase() also destroys the op, which recursively destroys its nested regions/operands — that part is O(size of the erased subtree), not O(1). The O(1) claim is specifically about unlinking from the parent Block.
~30 s.
-->

---

# 🧠 Quiz (5/7): how fast is the i-th `Attribute`?

What's the complexity of getting **the i-th `Attribute`**?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ Depends what "i-th" means

**Positionally**, `op->getAttrs()[i]` is **O(1)** — `DictionaryAttr`'s storage is a contiguous, name-**sorted** `ArrayRef<NamedAttribute>`, so raw indexing is plain array access.

But nobody asks for "the i-th attribute" in real code — the real operation is lookup **by name**:

```cpp
op->getAttr("answer")            // O(log n) — binary search over the sorted array
op->getAttrOfType<T>("answer")   // same, + a cast
```

That's **O(log n)**, not O(1) like a hash map — `DictionaryAttr` is a sorted array, not a hash table.

<!-- Speaker notes:
The trick: "i-th" and "by name" are different questions with different answers. Binary search is impl::findAttrSorted, used by DictionaryAttr::get(StringRef)/getNamed(...).
~45 s.
-->

---

# 🧠 Quiz (6/7): how fast is the i-th `Block` of a `Region`?

 What's the complexity of getting **the i-th `Block`** of one of its regions?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ O(n)

Getting the i-th `Block` of a `Region` is **O(n)**.

`Region::blocks` is `llvm::iplist<Block>`, a **doubly-linked list**: no random access, you must walk from the head.

<!-- Speaker notes:
~30 s.
-->

---

# 🧠 Quiz (7/7): how fast is the i-th `Operation*` in a `Block`?

What's the complexity of getting **the i-th `Operation*`** in one of its blocks?

<!-- Speaker notes:
~30 s.
-->

---

# ✅ O(n)

`Block::operations` is `llvm::iplist<Operation>`, a doubly-linked list.

**Note:** Accessing blocks / operations by index is extremely rare — as cases 3 and 4 showed, the IR data structures are optimized instead for splicing anywhere: `insert`, `erase`, `moveBefore` are all O(1).

<!-- Speaker notes:
Closing note for the whole 7-part quiz — land this explicitly if nothing else. Ties directly to the "Modification" cheat-sheet table, which is full of exactly these splice-anywhere operations.
~45 s.
-->

---

# ⏱ Zooming into `OpOperand` & `BlockOperand`

Both are edges, and both derive from the same base class:

```text
     ┌────────────────────────┐
     │     IROperandBase      │
     ├────────────────────────┤
     │ Operation *const owner │
     │ IROperandBase *nextUse │
     │ IROperandBase *back    │
     └────────────△───────────┘
       ┌──────────┴──────────┐
       │                     │
┌─────────────┐      ┌──────────────┐
│  OpOperand  │      │ BlockOperand │
├─────────────┤      ├──────────────┤
│ Value value │      │ Block *value │
└─────────────┘      └──────────────┘
```

- **`OpOperand`** — chains into the *value*'s use-list (its `firstUse`).
- **`BlockOperand`** — chains into the *target block*'s use-list (`firstUse`, from the previous slide).

<sub>mlir/include/mlir/IR/UseDefLists.h; Value.h (`OpOperand`); BlockSupport.h (`BlockOperand`)</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: OpOperand and BlockOperand both derive from IROperandBase (owner, nextUse, back) and each add exactly one field, `value` — one edge is "a Value I use", the other is "a Block I branch to" — and neither is separately allocated, they're just array elements trailing their owner Operation.
Simplification, flag if asked: `back` is drawn here as `IROperandBase *`, i.e. "the previous edge", for a clean doubly-linked-list mental model. The real field is `IROperandBase **back` — a pointer to the SLOT that currently points at this node (either the previous edge's `nextUse`, or the owning value/block's `firstUse`). That's what makes unlinking branch-free (head-of-list and middle-of-list removal are the exact same code) — the simplified "plain prev pointer" model would need an if-is-first-node special case instead. Good depth to have ready, not worth putting on the slide.
Also simplified: OpOperand/BlockOperand don't directly extend IROperandBase — there's a CRTP template layer in between, `IROperand<DerivedT, IRValueT>` (DerivedT=OpOperand/BlockOperand, IRValueT=Value/Block*), which is where `value` actually lives. Collapsed away here since it adds a name without adding a concept.
This slide is the payoff for three earlier promises: (1) the "Use-def chains" slide said "a use is an OpOperand — an edge that knows both endpoints" without showing fields, this is those fields; (2) the Operation memory-layout slide's OpOperand[]/BlockOperand[] rows, now explained; (3) the Block slide's firstUse row, now explained from the other end (BlockOperand's value points AT the Block, and getUseList chains into that Block's firstUse).
owner vs. value, the constant confusion point: owner = "whose operand list am I in" (e.g. the cf.br); value = "what am I pointing AT" (the Value being read, or the Block being branched to). getUsers()/getUses() on a Value walk the chain of OpOperands whose VALUE is that Value, across potentially many different OWNERS.
back is technically a pointer-to-pointer (IROperandBase**), not a direct "previous node" pointer — it points at whatever SLOT currently references this node (either another operand's nextUse, or the use-list head firstUse). That's what lets removeFromCurrent() unlink in O(1) without walking the list. "Doubly-linked list" is the right level of intuition for the slide; this is the precise mechanism if asked.
No stored index either: getOperandNumber() isn't a field, it's pointer arithmetic (this - &owner->getOpOperands()[0]) — only works because the OpOperand array is contiguous and trailing (payoff from the memory-layout slide).
Neither OpOperand nor BlockOperand is separately heap-allocated — both are plain array elements trailing their owner Operation's single allocation.
(+1 if presented.)
-->

---

# `Value`: exactly two forms

```text
┌────────────────────────────────┐          ┌─────────────────┐
│ IRObjectWithUseList<OpOperand> │          │      Value      │
├────────────────────────────────┤          ├─────────────────┤
│ IROperandBase *firstUse        │          │ ValueImpl *impl │
└────────────────△───────────────┘          └─────────────────┘
                 │                                   │
                 │                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                          ValueImpl                          │
├─────────────────────────────────────────────────────────────┤
│ PointerIntPair<Type,3,Kind> typeAndKind                     │
└──────────────────────────────△──────────────────────────────┘
                               │
                  ┌────────────┴──────────┐
                  │                       │
          ┌──────────────┐      ┌───────────────────┐
          │ OpResultImpl │      │ BlockArgumentImpl │
          └──────────────┘      ├───────────────────┤
                                │ Block *owner      │
                                │ int64_t index     │
                                │ Location loc      │
                                └───────────────────┘
```

<sub>mlir/include/mlir/IR/Value.h:40-84 (`ValueImpl`), 278-302 (`BlockArgumentImpl`), 353-380 (`OpResultImpl`), 454 (`OpResult`)</sub>

<!-- Speaker notes:
Answers two things students ask once they've seen the Operation memory-layout slides: "where's Value's superclass?" (nowhere — it's a bare handle, unlike Operation which is always heap+pointer) and "where's the uses-list pointer?" (on ValueImpl, inherited from IRObjectWithUseList<OpOperand> — same base class OpOperand's owner-side pointer chains into, from the earlier deep-dive).
OpResultImpl has zero extra fields of its own (box closes immediately) — result number comes from Kind for InlineOpResult, or a trailing outOfLineIndex field for OutOfLineOpResult (not shown here, that's the "what's really in memory" flex slide's territory). BlockArgumentImpl earns its 3 fields because block arguments need an explicit owner/index/loc that OpResult gets for free from its position in the trailing OpResultImpl array.
Don't dwell on OpResult/BlockArgument (the handle classes) here — one bullet is enough; they add no fields, so there's nothing to draw.
~2 min.
-->

---

# 🧠 Quiz: rewriting `x + x` into `x * 2`

Goal: rewrite `%r = arith.addi %x, %x` (same operand twice) into `%r = arith.muli %x, 2`. Find the bug(s):

```cpp
arith::AddIOp addOp = ...;
if (&addOp.getLhs() == &addOp.getRhs()) {
  OpBuilder b(addOp);
  Value two = arith::ConstantOp::create(
      b, addOp.getLoc(), b.getIntegerAttr(addOp.getLhs().getType(), 2));
  arith::MulIOp mulOp = arith::MulIOp::create(
      b, addOp.getLoc(), addOp.getLhs(), two);
  addOp->replaceAllUsesWith(mulOp->getResults());
  addOp->erase();
}
```

<!-- Speaker notes:
Give ~60-90 seconds, no separate answer slide. This is the original snippet a student submitted — every OTHER bug already got fixed for this quiz (there were several: undefined `b`, `ConstantOp::create` called with a `Type` and a raw `int` instead of `Location`+`Attribute`, an undefined `lhs` variable, `MulIOp::create` missing its `Location`, and — the sneaky one — erasing `mulOp` (the op we just made everyone use!) instead of `addOp` (the now-dead one)). Only the `if` condition is untouched; that's the whole quiz.
Answer: `&addOp.getLhs() == &addOp.getRhs()` **does not compile.** getLhs()/getRhs() return `Value` (well, `TypedValue<Type>`, still a Value) BY VALUE — a prvalue/temporary. You cannot apply the built-in unary `&` to a non-lvalue; `Value` doesn't overload `operator&` to accept one either. This is the same class of error as `int f(); &f();` — a hard compile error ("cannot take the address of an rvalue"), not a subtle runtime bug.
The fix (don't show it as "the answer" until asked — let them get there): drop the `&`s. `if (addOp.getLhs() == addOp.getRhs())` — Value::operator==(const Value&) (Value.h:101) is `impl == other.impl`, exactly the pointer-identity check "is this the same SSA value" that's wanted. This is the direct payoff of this slide's own bullet: Value is compared with ==, not &==&.
If someone protests "but what if it DID compile" — worth one sentence: even hypothetically, comparing addresses of two temporaries that are both alive in the same expression would be comparing two DIFFERENT stack slots (the compiler must materialize both to evaluate the ==), so it would be unconditionally false regardless of whether addOp.getLhs() and addOp.getRhs() are the same Value — doubly wrong, not just "wrong sometimes." But the real headline is simpler: it doesn't compile at all.
~2 min.
-->

---

# Handle discipline, the full picture

| Construct | Pass as |
|---|---|
| `Operation` | `Operation*` / `Operation&` |
| concrete op (`AddIOp`, ...) | **by value** |
| `Block` | `Block*` / `Block&` |
| `Region` | `Region*` / `Region&` |
| `Value` (`OpResult`/`BlockArgument`) | **by value** |
| `OpOperand` / `BlockOperand` | `OpOperand&` / `BlockOperand&` |
| `Attribute`, `Type`, `Location` | **by value** |

<!-- Speaker notes:
Your draft had operation/block/region/OpOperand-BlockOperand/Value/Attribute right; one addition:
`Type` and `Location` belong on the "by value" list too, same pattern as Attribute: both wrap exactly one interned pointer (Location literally wraps a LocationAttr, itself an Attribute), both copyable, both compared with plain pointer-equality on that one field. Anything of this shape (Value, Attribute, Type, Location, OperationName) is "cheap handle into the MLIRContext's uniquing tables" — one category, not four separate ones.
Also worth knowing (concrete ops): AddIOp etc. DO define == — a free function `operator==(OpState lhs, OpState rhs)` taking both by value and comparing `lhs.getOperation() == rhs.getOperation()`, so == on two typed ops means exactly what == on the underlying Operation* means.
Non-copyability isn't unique to OpOperand/BlockOperand — it's the general reason anything ends up in the pointer/reference row: Block explicitly deletes its copy ctor/assignment (Block.h), IROperandBase does the same for OpOperand/BlockOperand (UseDefLists.h:74-75), and Operation's constructor is private (only Operation::create can make one) — Region inherits non-copyability transitively via its owned Block list. The pattern: pointer/reference means "this holds real, non-trivial owned state that must never be copied"; by-value means "this is a cheap handle, copying it is free and safe."
One nuance if asked: OpOperand's own operator== ("this == &other") checks "is this literally the same slot/edge", a different question from "do these two edges point at the same Value" (that's `a.get() == b.get()`) — same family of mistake as the quiz two slides ago, just for OpOperand instead of Value.
~2 min.
-->

---

# Using a `Value`: `getOwner()`, `getDefiningOp()`

- `OpResult` → `getOwner()` returns the defining `Operation*`, `getResultNumber()`
- `BlockArgument` → `getOwner()` returns the `Block*`, `getArgNumber()`
- `val.getDefiningOp()` — the defining op, **or `nullptr` if it's a block argument** (function arguments included!)

```cpp
if (Operation *def = operand.getDefiningOp())
  llvm::outs() << "defined by " << def->getName() << "\n";
else  // must be a block argument
  llvm::outs() << "block arg #" << cast<BlockArgument>(operand).getArgNumber() << "\n";
```

**Handle discipline:** `Value` has value semantics (copy it freely, compare with `==`). `Operation` is pointer-only. Writing `Value*` or copying `Operation` — both wrong.

<!-- Speaker notes:
The getDefiningOp-returns-null gotcha is the #1 beginner segfault: they walk a function, call getDefiningOp on an operand that happens to be a function argument, and dereference nullptr. Drill it now: ALWAYS null-check, or use the templated form val.getDefiningOp<OpTy>() (shown later) which folds the null check in.
Also mention: BlockArgument and OpResult are subclasses of Value, and you can cast a Value with cast<BlockArgument>(v) — the LLVM casting machinery works on values too.
~2 min.
-->

---

# The navigation map ⏱

| I have… | I want… | Call |
|---|---|---|
| `Operation *op` | its parent | `op->getParentOp()`, `op->getParentOfType<func::FuncOp>()` |
| `Operation *op` | its block / region | `op->getBlock()`, `op->getParentRegion()` |
| `Operation *op` | what's nested inside | `op->getRegions()` → `region.getBlocks()` → `block.getOperations()` |
| `Block &b` | only ops of one type | `b.getOps<arith::ConstantOp>()` |
| `Operation *op` | inputs / outputs | `op->getOperand(i)`, `op->getOperands()`, `op->getResult(i)`, `op->getResults()` |
| `Operation *op` | attributes | `op->getAttrs()`, `op->getAttrOfType<IntegerAttr>("answer")` |
| `Operation *op` | context / location | `op->getContext()`, `op->getLoc()` |
| `Value v` | who defines it | `v.getDefiningOp()` *(null for block args)* |
| `Value v` | who uses it | `v.getUsers()`, `v.getUses()` |

Print anything while debugging: `op->dump()`, or stream it: `llvm::errs() << *op << "\n";`

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: this table is your navigation reference card for the exercise — it reappears on the cheat sheet at the end, use it from there.
Don't read the table aloud — tell them this slide is their reference card for the exercise (it reappears in the cheat sheet at the end). Walk through just two rows: op->getParentOfType<func::FuncOp>() ("climb until you hit a function") and block.getOps<T>() ("filtered iteration over one block, no casting needed").
op->dump() prints to stderr with a trailing newline — the debugging workhorse, works in a debugger too ("call op->dump()" in gdb/lldb).
(+1 if presented.)
-->

---

# Two views of one op: generic vs. generated

The same in-memory `Operation` node can be looked at through two lenses:

- **Generic lens** — `Operation *`: uniform API, works for *any* op, even unregistered ones. `getOperand(0)`, `getNumResults()`, `getAttrs()`.
- **Typed lens** — the ODS-generated class, e.g. `arith::AddIOp`: a thin **value-semantic wrapper around the same `Operation*`** that adds named accessors.

```cpp
arith::AddIOp add = ...;
Value lhs = add.getLhs();      // named accessor, from ODS arg name "lhs"
Value lhs2 = add->getOperand(0); // identical value, generic spelling
Operation *raw = add.getOperation(); // unwrap the lens
```

- ODS argument names → accessors: `$lhs` → `getLhs()`, plus `getLhsMutable()` to reassign the operand in place.
- The `->` operator on a typed op forwards to the underlying `Operation`, so *everything* from the previous slides works on typed ops too.

**When to use which?** Generic for op-agnostic code (walking, printing, moving); typed the moment you care what the op *means*.

<!-- Speaker notes:
The mental model to repeat: the typed op class is a LENS, not a different object. arith::AddIOp is one pointer wide — it holds the Operation* and nothing else. Creating/copying op classes is free.
Connect to their ODS session: the accessors come from the names THEY wrote in the .td arguments list. This is where TableGen output meets hand-written C++.
~2 min.
-->

---

# Switching lenses: `isa` / `dyn_cast` / `cast`

MLIR uses LLVM's casting infrastructure (not C++ RTTI — MLIR builds with `-fno-rtti`, `dynamic_cast` won't even compile):

```cpp
void inspect(Operation *op) {
  // Generic API: works for ANY op, registered or not.
  llvm::outs() << op->getName().getStringRef() << " has "
               << op->getNumOperands() << " operands, "
               << op->getNumResults() << " results\n";
  for (NamedAttribute attr : op->getAttrs())
    llvm::outs() << "  attr " << attr.getName() << " = " << attr.getValue() << "\n";

  // Typed API: cast to the ODS-generated class for named accessors.
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    Value lhs = add.getLhs();    // == op->getOperand(0)
    Value res = add.getResult(); // == op->getResult(0)
  }
}
```

| Call | Behavior |
|---|---|
| `isa<AddIOp>(op)` | `bool` — is it one? |
| `dyn_cast<AddIOp>(op)` | typed op, or *null op* on mismatch — **the workhorse** |
| `cast<AddIOp>(op)` | asserts on mismatch — only when you *know* |
| `dyn_cast_or_null<T>(x)` | like `dyn_cast`, but also tolerates null input |

<!-- Speaker notes:
Two gotchas to say out loud:
1. isa/dyn_cast on a NULL Operation* crashes (for op casts it's a plain null dereference — MLIR's cast machinery bypasses LLVM's null assert); dyn_cast_or_null / isa_and_nonnull are the null-tolerant forms. Don't elaborate further — the flex quiz right after this slide delivers this gotcha as a spot-the-crash with a real upstream commit. If you're SKIPPING that quiz, say the punchline now: getDefiningOp returns null for block arguments, and val.getDefiningOp<OpTy>() folds the null check in.
2. op->getName() returns an OperationName object, not a string. Streaming it works; comparing needs getName().getStringRef() — but if you're comparing op names as strings, you almost always want isa<> instead.
The inspect() snippet is verified to compile against this checkout.
~2 min. Core-path check: leaving this slide you should be ~9 min into the lecture (≈0:14 on the session clock).
-->

---

# 🧠 Quiz: this helper shipped upstream ⏱

Condensed from a real conversion pass — this ran in production:

```cpp
/// Returns true iff the extension op is fed by a vector.transfer_read.
static bool isFedByTransferRead(arith::ExtUIOp extOp) {
  return isa<vector::TransferReadOp>(extOp.getOperand().getDefiningOp());
}
```

All of the pass's tests passed. Then a user's kernel contained this:

```mlir
func.func @f(%v: vector<4xi8>) -> vector<4xi32> {
  %0 = arith.extui %v : vector<4xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}
```

*(`vector.transfer_read` = a vector load — its details don't matter here.)*

What does the helper do for `%0`'s op?

① returns `false` — no `transfer_read` anywhere &nbsp;&nbsp; ② returns `true`
③ crash / segfault &nbsp;&nbsp; ④ verifier error after the pass

<sub>pre-fix code of mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp (condensed); the fix is quoted on the next slide</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip together with the answers slide that follows); if skipped, say: isa/dyn_cast on getDefiningOp() crashes on block arguments — always reach for the null-safe templated v.getDefiningOp<OpTy>(); this exact crash shipped upstream and keeps being re-fixed across dialects.
Give ~60 seconds, vote by show of hands. ④ is for students who think the verifier catches everything; ① for those who missed that getDefiningOp can return null.
Answer: ③. %v is a BlockArgument, so getDefiningOp() returns nullptr (the fine print from the Value slide) — and isa<> on a null Operation* is a plain null dereference: segfault, in EVERY build type. Careful with the mechanism: LLVM's famous "isa<> used on a null pointer" assert does NOT fire here — MLIR's CastInfo<T, Operation*> specialization (mlir/include/mlir/IR/Operation.h:1187-1191) calls OpTy::classof(op) directly and bypasses it; the crash lands in Op::classof → getRegisteredInfo dereferencing null (verified by compiling the buggy helper against this tree: a raw null read inside Operation::getName, no assert; the upstream issue #107967 stack trace is likewise a raw SIGSEGV).
Framing fine print: in the real pass this helper only ran on extension ops found in a vector.contract's backward slice, so the 3-line function alone wouldn't reach it — hence "a user's kernel contained this", not "ran the pass on this file". The quiz question asks what the HELPER does for %0's op, which is exact.
Provenance (verified in this checkout's history): commit 927559d27d5b "[mlir][vector] Fix a crash in VectorToGPU (#113454)" removed exactly this isa<...>(getDefiningOp()) call; the commit message notes the operand "cannot be retrieved using getDefiningOp" when it is a function argument (issue #107967). The same class keeps being re-fixed independently: f6a756f35a4d (#108703, linalg isContractionBody segfault), 96aef1a11382 (#195150, linalg FoldAddIntoDest) — arguably the most re-fixed crash in MLIR pattern code.
Why tests never caught it: every test fed the extui from another op. Values without defining ops (function args, loop iter_args) only appear in fuller IR — the first such input in the wild took the compiler down.
(+2 if presented.)
-->

---

# ✅ It crashes (③) — the null nobody tested ⏱

- `%v` is a **block argument** — `getDefiningOp()` returns **`nullptr`** (that fine print again).
- `isa<>` on a null `Operation*` dereferences it: **segfault — in every build type.** (MLIR's op-cast machinery calls `classof` directly, *bypassing* LLVM's famous `"isa<> used on a null pointer"` assert. Not even an assertions build saves you.)
- Every test fed the `extui` from another op; the first **function argument** in the wild took the whole compiler down.

The real fix is one line — the templated form folds in the null check (`dyn_cast_or_null` under the hood):

```cpp
static bool isFedByTransferRead(arith::ExtUIOp extOp) {
  // Typed + null-safe in one call:
  auto read = extOp.getOperand().getDefiningOp<vector::TransferReadOp>();
  return read != nullptr;  // null: block argument OR a different producer
}
```

Values with no defining op are everywhere: **function arguments, `scf.for` induction variables, `iter_args`**. Every backward step through a use-def chain must survive them.

<sub>fix: mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp (#113454); templated overload: mlir/include/mlir/IR/Value.h:124-127; op-cast null bypass: mlir/include/mlir/IR/Operation.h:1187</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; paired with the quiz slide before it — skip or present the pair together.
Reinforce the idiom hierarchy: (1) v.getDefiningOp<OpTy>() when you want a specific op type — null-safe, one call; (2) plain getDefiningOp() + explicit null check when you need the generic Operation*; (3) isa_and_nonnull<T>(v.getDefiningOp()) when you only need the bool.
Callback: "Switching lenses" gotcha #1 warned about null op pointers — this is that gotcha with a commit hash attached.
Exercise tie-in: checkpoint 3 of Exercise 1 is exactly this kind of robustness. Their matchPattern-based solution is naturally safe (matchPattern handles block arguments); hand-rolled getDefiningOp chains are where this bites.
(+1 if presented.)
-->

---

# Use-def chains: the wires between ops

SSA in memory: every `Value` keeps a **list of its uses**. A *use* is an `OpOperand` — an edge that knows both endpoints:

```text
          defines                      operand #0
  ┌────────────┐   use-list   ┌─────────────────────┐
  │ %x = addi  │ ───────────► │ OpOperand            │──► owner: muli
  └────────────┘        │     └─────────────────────┘
                        │     ┌─────────────────────┐
                        └───► │ OpOperand            │──► owner: return
                              └─────────────────────┘
```

```cpp
for (Operation *user : x.getUsers())       // the ops using %x
  llvm::outs() << "used by " << user->getName() << "\n";
for (OpOperand &use : x.getUses())         // the edges: owner + operand index
  llvm::outs() << "operand #" << use.getOperandNumber()
               << " of " << use.getOwner()->getName() << "\n";
```

Cheap queries: `x.use_empty()`, `x.hasOneUse()`, `x.hasNUses(n)`.
**Trap:** `x.getNumUses()` walks the whole list — *linear time* (documented!). Don't count when you can ask.

<!-- Speaker notes:
The two directions to keep straight: operands point "up" to values (data flowing in); the use-list points "down" from a value to everyone consuming it (this is the extra bookkeeping SSA buys you — you can find all users in O(uses)).
getUsers() vs getUses(): users gives Operation*, uses gives the edges (OpOperand&) — you need the edge when you care WHICH operand slot uses the value, or want to reassign just that slot (use.set(newValue)).
Subtle gotcha: getUsers() can yield the same op twice if it uses the value in two operands (muli %x, %x). It iterates uses, not a deduplicated set.
Linear getNumUses is at mlir/include/mlir/IR/Value.h:189-193.
~2 min.
-->

---

# Rewiring: `replaceAllUsesWith` and friends

“RAUW” = replace all uses with. Three flavors, one inverse:

```cpp
// 1. Value-level: every use of one SSA value now uses another.
oldVal.replaceAllUsesWith(newVal);

// 2. Conditional: only the uses your predicate accepts.
oldVal.replaceUsesWithIf(newVal, [&](OpOperand &use) {
  return use.getOwner()->getBlock() == someBlock;
});

// 3. Operation-level: all results at once — then the op is dead.
oldOp->replaceAllUsesWith(newOp->getResults());
oldOp->erase();   // safe ONLY once use_empty()

// Inverse direction: rewrite ONE op's own operands.
userOp->replaceUsesOfWith(/*from=*/oldVal, /*to=*/newVal);
```

**Two directions — don't mix them up:**
- `replaceAllUsesWith` edits *other ops'* operands (everyone pointing at me).
- `replaceUsesOfWith` edits *this op's* operands (what I point at).

**Iron rule: RAUW first, `erase()` second.** Erasing an op whose results still have uses is a fatal error in assert builds (`"operation destroyed but still has uses"`) — and silent memory corruption in release builds.

<!-- Speaker notes:
This slide is the heart of IR surgery. The RAUW-then-erase ordering rule will come back in the spot-the-bug quiz.
The fatal error is in Operation's destructor (mlir/lib/IR/Operation.cpp:176-186) — in debug builds it even prints each offending user before dying. Show appreciation for assert builds: this is why we teach on an assertions-enabled tree.
Fine print: op->replaceAllUsesWith is a template over "values"; for a single Value wrap it — op->replaceAllUsesWith(ValueRange{v}) — or go through op->getResult(0).replaceAllUsesWith(v). Mismatched result counts assert at runtime.
~2 min.
-->

---

# 🧠 Quiz: after the RAUW

```mlir
func.func @quiz(%a: i32, %b: i32) -> i32 {
  %x = arith.addi %a, %b : i32
  %y = arith.subi %a, %b : i32
  %u = arith.muli %x, %x : i32
  %v = arith.addi %u, %y : i32
  return %v : i32
}
```

Suppose a pass runs: `x.replaceAllUsesWith(y);` *(where `x`, `y` are the C++ `Value`s for `%x`, `%y`)*

1. Which operations had their operands changed?
2. Can we now safely call `erase()` on the op defining `%x`?
3. Can we safely `erase()` the op defining `%y`?
4. What does `y.getUsers()` yield now?

<!-- Speaker notes:
Give ~90 seconds of think time, collect answers by show of hands per question.
Answers (next slide has them too):
1. Exactly ONE op changed: the muli — both of its operand slots (%x, %x) now hold %y. Nothing else used %x. %v is untouched: it uses %u and %y, and RAUW never touches the definition of %y.
2. YES — %x is now use_empty(), so erasing the addi is legal.
3. NO — %y now has three uses (muli twice + the final addi). Erase would hit the "still has uses" fatal error.
4. Three entries: muli, muli, addi — getUsers() iterates USES, so an op using the value twice appears twice. If someone answers "two ops", that's the teachable moment.
~2 min with discussion.
-->

---

# ✅ RAUW quiz answers

```mlir
  %x = arith.addi %a, %b : i32   // ← now use_empty(): erasable ✔
  %y = arith.subi %a, %b : i32   // ← now has THREE uses: NOT erasable ✘
  %u = arith.muli %y, %y : i32   // ← the only op that changed (both slots)
  %v = arith.addi %u, %y : i32   // ← unchanged (never used %x)
  return %v : i32
```

1. **Only the `muli` changed** — RAUW rewrites the *uses* of `%x`, nothing else.
2. **Yes** — `%x` is `use_empty()`; `erase()` is safe.
3. **No** — `%y` now has 3 uses; erasing its definer would be the fatal `"operation destroyed but still has uses"`.
4. `y.getUsers()` yields **`muli`, `muli`, `addi`** — one entry *per use*, so `muli` appears twice.

<!-- Speaker notes:
Reinforce: RAUW is O(uses of %x) — it splices the use-list, it doesn't scan the function. That's what the use-list bookkeeping buys.
If time allows, foreshadow: "erase the addi defining %x" is exactly what DCE does automatically for side-effect-free ops — Session 3.
~1 min. Core-path check: ~16 min into the lecture (≈0:21).
-->

---

# Traversal, level 0: follow the nesting by hand ⏱

The object model gives you the loops for free (this is the core of the real `-test-print-nesting` pass):

```cpp
void printOperation(Operation *op) {
  llvm::outs() << "op: " << op->getName() << "\n";
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())   // Region is iterable too
      for (Operation &nested : block)          // == block.getOperations()
        printOperation(&nested);
}

// Filtered, one nesting level only:
for (arith::ConstantOp c : block.getOps<arith::ConstantOp>())
  llvm::outs() << "constant: " << c.getValue() << "\n";
```

Fine for one level. Tedious for "find every `muli` anywhere in this function" — regions nest arbitrarily deep (`scf.for` inside `scf.if` inside …).

<sub>adapted from mlir/test/lib/IR/TestPrintNesting.cpp:32-74</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: you can hand-roll nested for-loops over regions/blocks/ops — that's what -test-print-nesting does — but walk() on the next slide is what you'll actually write.
Point out this is real upstream code (the test pass we'll run in the demo shortly), just condensed. The recursion mirrors the ownership diagram from earlier — nothing new, that's the point.
Then set up the pain: any interesting matcher needs to see through nesting, and hand-rolled recursion everywhere is noise. Enter walk().
(+1 if presented.)
-->

---

# Traversal, level 1: `walk()`

`walk` = “call my callback on every op nested under (and including!) this one”:

```cpp
// Every op, post-order (children before parents), lexical order in blocks:
root->walk([](Operation *op) { llvm::outs() << op->getName() << "\n"; });

// Typed filtering — the callback's parameter type selects the ops:
root->walk([](arith::MulIOp op) { /* only muli ops, already typed! */ });

// Parents before children instead:
root->walk<WalkOrder::PreOrder>([](Operation *op) { /* ... */ });
```

- Default order is **post-order** — by the time you see an op, its regions are done.
- The typed-callback filter is not a cast — it *silently skips* every non-matching op. Interfaces work too: `walk([](LoopLikeOpInterface loop) {...})`.
- `Block::walk` and `Region::walk` exist as well.

<!-- Speaker notes:
Two things students misread:
1. walk visits the root itself ("nested under AND INCLUDING") — a walk started at a ModuleOp calls the callback on the module too. Surprising when the callback assumes ops with a parent.
2. The typed parameter is a FILTER, not a cast — beginners think it's sugar for cast<> and are confused why "nothing happens" for other ops. It's implemented with dyn_cast internally; non-matching ops are skipped.
Post-order default is documented in Operation.h (walk template default WalkOrder::PostOrder, Operation.h:817).
~2 min.
-->

---

# Controlling a walk: `WalkResult` ⏱

Return `WalkResult` from the callback to steer the traversal — `advance` continues, `interrupt` aborts everything, `skip` prunes the subtree:

```cpp
WalkResult result = root->walk([](Operation *op) {
  if (isa<gpu::LaunchOp>(op))
    return WalkResult::interrupt();  // abort the entire walk
  if (isa<scf::ForOp>(op))
    return WalkResult::skip();       // don't descend into loops
  return WalkResult::advance();      // keep going (the default)
});
if (result.wasInterrupted()) { /* found one */ }
```

- A `void` callback = always advance.
- `WalkResult` implicitly converts from `LogicalResult` (failure ⇒ interrupt) — handy for verifier-style walks.

<sub>mlir/include/mlir/Support/WalkResult.h:29-55</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the walk callback can return WalkResult::interrupt/skip/advance to abort or prune the traversal — look this slide up when you need it.
interrupt vs skip in one sentence: interrupt = "I'm done with the whole walk" (e.g. found what I searched for); skip = "don't look inside THIS op, continue with its siblings" (e.g. don't descend into this loop).
Note skip only makes sense in pre-order walks — in post-order the nested ops were already visited by the time the callback runs on the parent.
(+1 if presented.)
-->

---

# Mutating during a walk: the rules

The documented guarantee (Operation.h): the callback may **erase the op it is currently visiting** if

- the walk is **post-order** (the default), **or**
- the walk is pre-order **and** you return `WalkResult::skip()` after erasing.

Everything else — erasing *other* not-yet-visited ops, erasing your parent — invalidates the iteration. Undefined behavior, often "works" until it doesn't.

**The robust beginner idiom: collect, then mutate.**

```cpp
SmallVector<arith::MulIOp> candidates;
getOperation()->walk([&](arith::MulIOp op) { candidates.push_back(op); });

for (arith::MulIOp op : candidates) {
  // ... create replacement, RAUW, op->erase() — no live iterator to break
}
```

Two phases, no aliasing between iteration and mutation. Costs one `SmallVector`; saves one afternoon of debugging.

<sub>guarantee: mlir/include/mlir/IR/Operation.h:796-799</sub>

<!-- Speaker notes:
Why the guarantee holds: the walk implementation iterates blocks with an early-increment iterator — it steps PAST the current op before invoking the callback, so deleting the current op is fine, but the "next" pointer already taken can be invalidated by deleting other ops.
Message to hammer: when in doubt, collect-then-mutate. It also separates "what to change" from "how to change it", which makes the code reviewable. The exercise starter nudges toward this shape.
Ops you CREATE during a walk before the current position are fine — they simply won't be visited (post-order already passed them).
~2 min.
-->

---

# 🔴 Live demo: seeing the structure with a real tool

```bash
build/bin/mlir-opt -test-print-nesting -allow-unregistered-dialect \
    mlir/test/IR/print-ir-nesting.mlir -o /dev/null
```

```text
visiting op: 'builtin.module' with 0 operands and 0 results
 1 nested regions:
  Region with 1 blocks:
    Block with 0 arguments, 0 successors, and 2 operations
      visiting op: 'dialect.op1' with 0 operands and 4 results
      1 attributes:
       - 'attribute name' : '42 : i32'
       0 nested regions:
      visiting op: 'dialect.op2' with 0 operands and 0 results
      ...
        Region with 3 blocks:
          Block with 0 arguments, 2 successors, and 2 operations
```

<!-- Speaker notes:
Run it live from the llvm-project checkout root; output above is the real (trimmed) output of this exact command on this tree.
Narrate the correspondence: this text is printOperation/printRegion/printBlock from the traversal slides, executing. The source is mlir/test/lib/IR/TestPrintNesting.cpp — genuinely readable, send them there.
One-line mention, don't run it: the sibling pass --test-print-defuse prints producer/consumer info for every value (mlir/test/lib/IR/TestPrintDefUse.cpp).
Note the -o /dev/null: these passes only print; we discard the IR output.
These test-* flags exist because mlir-opt is built with MLIR_INCLUDE_TESTS=ON (in-tree default); they're not in libMLIR.
~2 min. Core-path check: ~22 min into the lecture (≈0:27).
-->

---

# `OpBuilder`: a cursor into the IR

To create ops you need two things: an `MLIRContext` and a **place**. `OpBuilder` bundles both — it's a cursor (`Block*` + position) where every newly created op is inserted:

```cpp
OpBuilder b(ctx);                    // ⚠ NO insertion point yet!
b.setInsertionPointToStart(&block);  // cursor: start of block
b.setInsertionPoint(op);             // cursor: right BEFORE op
b.setInsertionPointAfter(op);        // cursor: right after op

OpBuilder before(op);                // shorthand: constructor == setInsertionPoint(op)
OpBuilder atEnd = OpBuilder::atBlockEnd(&block);
```

**Gotcha #1:** `OpBuilder b(op)` inserts **BEFORE** `op` — not after. (Perfect when replacing `op`; surprising otherwise.)

**Gotcha #2:** `OpBuilder b(ctx)` alone has no insertion point. Creating an op then does **not** crash — the op is built *detached*, inserted nowhere: it silently never appears in your output (and leaks). The docs call this an error; the API won't stop you.

<sub>mlir/include/mlir/IR/Builders.h:215-258, 400-440</sub>

<!-- Speaker notes:
Analogy that lands: the builder is a text-editor cursor. Constructors and setInsertionPoint* move the cursor; creating an op types at the cursor (and the cursor advances past what was typed, so consecutive creates appear in order).
Gotcha #2 detail (verified on this tree): OpBuilder::insert does `if (block) ...` and silently skips insertion when no insertion point is set (mlir/lib/IR/Builders.cpp, OpBuilder::insert) — no assert fires, even in debug builds. The op is simply detached. Do NOT tell students it crashes; the danger is precisely that it doesn't.
The BEFORE-not-after choice makes sense once you see the replacement idiom: new ops go before the op being replaced so its uses can be redirected to them.
~2 min.
-->

---

# Saving your place: `InsertionGuard` ⏱

Moving the cursor deep into a region and forgetting to move it back is the classic “my ops ended up in the wrong block” bug. RAII to the rescue:

```cpp
{
  OpBuilder::InsertionGuard guard(b);   // saves the current insertion point
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(42));
} // cursor restored here, automatically
```

- Use it whenever a helper function temporarily redirects a builder it doesn't own.
- Nested guards nest correctly (it's just save/restore).

<sub>mlir/include/mlir/IR/Builders.h:350-376</sub>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: OpBuilder::InsertionGuard is RAII save/restore for the insertion point — open any helper that moves a borrowed builder's cursor with one; it's on the cheat sheet.
Short slide — one habit to install: any function that takes OpBuilder& and moves the insertion point should open with an InsertionGuard, unless moving the point IS its contract.
(+1 if presented.)
-->

---

# Creating ops: `OpTy::create(builder, loc, ...)`

Every ODS op gets static `create` methods — one per `build(...)` overload you saw in the ODS session:

```cpp
OpBuilder b(op);                    // insert before the op we're replacing
Location loc = op.getLoc();         // reuse its location (more in a second)

Value shift = arith::ConstantOp::create(
    b, loc, b.getIntegerAttr(op.getType(), 3));
Value shl = arith::ShLIOp::create(b, loc, op.getLhs(), shift);
```

> **Heads-up for old code:** you will see `builder.create<arith::ShLIOp>(loc, ...)` in blog posts, pre-2025 tutorials, and LLM output. That spelling is **deprecated** (since Oct 2025) — same behavior, but new code must use `OpTy::create(builder, loc, ...)`. Mentally rewrite the old form when you see it.

<sub>deprecation: mlir/include/mlir/IR/Builders.h:506-508</sub>

<!-- Speaker notes:
This deck mentions the deprecation exactly once — here. If students copy-paste the old form they'll get -Wdeprecated warnings, not errors, so tell them to treat the warning as a stop sign.
The create call inserts at the builder's cursor AND returns the typed op; assigning to Value implicitly takes result 0 (single-result ops convert to their result Value).
~1 min.
-->

---

# Constants, attributes, and `Location` discipline ⏱

`OpBuilder` inherits a bag of factory helpers from `Builder` (types, attributes, locations):

```cpp
IntegerAttr a  = b.getI32IntegerAttr(7);          // 7 : i32
IntegerAttr ix = b.getIndexAttr(4);               // 4 : index
Type i32       = b.getI32Type();
IntegerAttr c  = b.getIntegerAttr(someType, 3);   // typed constant payload

// A constant op is: attribute payload + arith.constant
Value cst = arith::ConstantOp::create(b, loc, b.getIntegerAttr(ty, 3));
```

**Location discipline.** Every op carries a `Location` (diagnostics + debug info survive on it):

- When replacing an op, **reuse `op.getLoc()`** for everything you create in its place.
- `b.getUnknownLoc()` is the last resort — an op with unknown location can't point users anywhere when a diagnostic fires.

One-liner to know: `b.createOrFold<OpTy>(loc, ...)` — create the op *and immediately try to fold it*; returns a `Value` that may be a pre-existing constant instead of a new op. (Folding is Session 3's topic.)

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the builder factory helpers (getIntegerAttr & co.) appear in the worked example in a minute, and the one location rule is mechanical — new ops that replace an old op reuse op.getLoc().
Locations feel like bureaucracy until the first time a verifier error says "unknown location" and you have 40k ops. Make the rule mechanical: new op replaces old op => new op gets old op's loc. There are FusedLoc/NameLoc for merging several source ops into one result op — mention only if asked.
createOrFold: don't deep-dive; plant the name so Session 3 can pick it up ("who folded my op? you asked for it").
(+1 if presented.) The worked-example section is next — it's the payoff, give it room.
-->

---

# 🧠 Quiz: spot the bug

Goal: rewrite `muli %x, C` (C a power of two) into `shli`. Does this work?

```cpp
void runOnOperation() override {
  getOperation()->walk([&](arith::MulIOp op) {
    APInt rhs;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&rhs)) || !rhs.isPowerOf2())
      return;
    OpBuilder b(op);
    Value shift = arith::ConstantOp::create(
        b, op.getLoc(), b.getIntegerAttr(op.getType(), rhs.logBase2()));
    Value shl = arith::ShLIOp::create(b, op.getLoc(), op.getLhs(), shift);
    op->erase();                              // (A)
    op->replaceAllUsesWith(ValueRange{shl});  // (B)
  });
}
```

Three candidate concerns — which are real bugs?
① The erase/RAUW order (A)/(B) &nbsp; ② erasing `op` *inside* the walk &nbsp; ③ creating new ops *inside* the walk

<sub>First use: `matchPattern(v, m_ConstantInt(&rhs))` = "is `v` a constant integer? bind its value into `rhs`" — details on a later slide.</sub>

<!-- Speaker notes:
Give ~90 seconds. matchPattern/m_ConstantInt is glossed in the on-slide footnote (read it aloud so the quiz is self-contained) and explained fully on the worked-example slide.
Answers:
① REAL BUG, twice over. erase() while the result still has uses = fatal "operation destroyed but still has uses" in assert builds / memory corruption in release. And line (B) then calls a method on freed memory — use-after-free. Correct order: RAUW, THEN erase.
② NOT a bug — erasing the currently-visited op in a (default) post-order walk is explicitly documented as allowed (Operation.h:796-799). But fragile under refactoring (someone flips it to pre-order…), which is why the model solution collects first.
③ NOT a bug — the new ops are inserted before the current op; a post-order walk has already moved past that position, they simply won't be visited.
Poll each concern separately with a show of hands — ② splits the room nicely.
~2 min including discussion.
-->

---

# ✅ Spot-the-bug answers

**① is the bug** — and it's two bugs in one:

- `erase()` while the `muli`'s result still has uses → fatal error in assert builds:
  `"operation destroyed but still has uses"` (release builds: silent corruption).
- Line (B) then touches the *freed* op — use-after-free.

**Fix: swap the two lines.** RAUW first, erase second. Always.

**② is legal** — a post-order walk may erase the *currently visited* op (documented guarantee). Erasing *other* ops during the walk would be the bug.

**③ is fine** — new ops inserted before the current op are simply never visited by the ongoing post-order walk.

Still, the model solution uses **collect-then-mutate**: it doesn't depend on walk-order fine print, and it survives refactoring.

<!-- Speaker notes:
Repeat the iron rule slowly: RAUW, then erase. It's the single most common crash in exercise submissions.
Transition: "now let's see the correct version — in full, and running."
~1 min.
-->

---

# Worked example: `MulToShift`, the real thing

```cpp
void runOnOperation() override {
  // Collect first, mutate after: don't erase while iterating.
  SmallVector<arith::MulIOp> candidates;
  getOperation()->walk([&](arith::MulIOp op) { candidates.push_back(op); });

  for (arith::MulIOp op : candidates) {
    APInt rhs;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&rhs)) || !rhs.isPowerOf2())
      continue;
    OpBuilder b(op); // insertion point: right before `op`
    Value shift = arith::ConstantOp::create(
        b, op.getLoc(), b.getIntegerAttr(op.getType(), rhs.logBase2()));
    Value shl = arith::ShLIOp::create(b, op.getLoc(), op.getLhs(), shift);
    op->replaceAllUsesWith(ValueRange{shl}); // RAUW
    op->erase();                             // now safe: use_empty()
  }
}
```

- `matchPattern(value, m_ConstantInt(&rhs))` — "is this value a constant integer? if so, bind it into `rhs`" (an `APInt`). Matches *any* constant-like op, not just `arith.constant`.
- `rhs.isPowerOf2()`, `rhs.logBase2()` — `APInt` does the math.

<!-- Speaker notes:
This exact code is compiled and run against this checkout (as an mlir-opt pass plugin) — the next slide shows its real output. It is also, deliberately, the shape of Exercise 1: walk+collect (checkpoint 1), match, build, RAUW, erase (checkpoint 2), and the guards that make it robust (checkpoint 3).
Why matchPattern instead of dyn_cast<arith::ConstantOp>? It sees through anything ConstantLike (other dialects' constants, splat constants) — the idiomatic upstream way to test for constants (mlir/include/mlir/IR/Matchers.h).
Walk through the data flow once: candidates -> guard -> builder before op -> two new ops -> RAUW -> erase. Every line uses an API from the last 20 slides. Nothing new except matchPattern (glossed on the quiz slide already).
~2 min.
-->

---

# 🔴 Live demo: run it

```bash
build/bin/mlir-opt mul.mlir --load-pass-plugin=./mul_to_shift.so \
    --pass-pipeline='builtin.module(func.func(mul-to-shift))'
```

<div class="columns">
<div>

**Before** (`mul.mlir`)

```mlir
func.func @mul_by_8(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %r = arith.muli %x, %c8 : i32
  return %r : i32
}

func.func @mul_by_7(%x: i32) -> i32 {
  %c7 = arith.constant 7 : i32
  %r = arith.muli %x, %c7 : i32
  return %r : i32
}
```

</div>
<div>

**After** (real output)

```mlir
func.func @mul_by_8(%arg0: i32) -> i32 {
  %c8_i32 = arith.constant 8 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.shli %arg0, %c3_i32 : i32
  return %0 : i32
}

func.func @mul_by_7(%arg0: i32) -> i32 {
  %c7_i32 = arith.constant 7 : i32
  %0 = arith.muli %arg0, %c7_i32 : i32
  return %0 : i32
}
```

</div>
</div>

`@mul_by_7` untouched (7 isn't a power of two) — but look at `@mul_by_8`: **`%c8_i32` is now dead.** We never erased it. Who cleans that up? → Session 3. (Spoiler: append `,canonicalize` to the pipeline.)

<!-- Speaker notes:
Demo prep (do before class; total ~30s): compile the pass as a plugin so stock mlir-opt can run it —
  c++ -shared -fPIC -fno-rtti -std=c++17 -O1 \
    -I mlir/include -I build/tools/mlir/include -I llvm/include -I build/include \
    mul_to_shift.cpp -o mul_to_shift.so
(mul_to_shift.cpp = the runOnOperation from the previous slide wrapped in a PassWrapper + mlirGetPassPluginInfo; keep that boilerplate off-slide — students get the pass pre-registered in school-opt and never need plugins.)
The "After" pane is the exact mlir-opt output of this command on this machine (module wrapper trimmed).
Things to narrate: (1) %x became %arg0 — the printer regenerates value names; names are display-only, not identity; (2) the shift constant %c3_i32 landed right before the shli — that's OpBuilder b(op) inserting BEFORE the muli; (3) THE DEAD %c8_i32 — we replaced+erased the muli but nobody asked us to erase the constant. Run the punchline live:
  build/bin/mlir-opt mul.mlir --load-pass-plugin=./mul_to_shift.so \
    --pass-pipeline='builtin.module(func.func(mul-to-shift,canonicalize))'
→ the dead constant vanishes (verified). One sentence: "canonicalize includes dead-code elimination for side-effect-free ops — Session 3 explains exactly why and when."
~3 min. Core-path check: ~33 min into the lecture (≈0:38). If over, skip the canonicalize punchline run and just tell it — the print-ir-after-all demo at the end reuses this exact run and shows the same story.
-->

---

# From snippet to pass: what's still missing

Our `runOnOperation` needs a home. A **pass** needs:

- a **name** and a **CLI flag** (`-mul-to-shift`)
- an **anchor**: which op does it run on? (`func.func` for us)
- **registration** so `mlir-opt` / `school-opt` can find it
- plumbing: `clonePass`, options, statistics, …

You write **none of that by hand.** The modern recipe:

```text
Passes.td                 TableGen definition (flag, anchor, docs, options)
   │  mlir-tblgen -gen-pass-decls
   ▼
Passes.h.inc              generated: impl::<Name>Base<Derived> CRTP base class
   │  you derive from it
   ▼
YourPass.cpp              #define GEN_PASS_DEF_<NAME> + runOnOperation()
```

<!-- Speaker notes:
Frame it as: the pass *body* is what you saw; the pass *identity* is declarative. Same philosophy as ODS for ops — declare in TableGen, generate the boilerplate, implement only the interesting method.
Historical note if asked: you'll find PassWrapper-based passes in test code and old tutorials — that's the manual non-TableGen path, only for quick hacks (the demo plugin uses it for compactness).
~1 min.
-->

---

# The smallest real pass: `-strip-debuginfo`

The **entire** implementation, from upstream (license, includes, `using namespace mlir;`, and one inner loop trimmed):

```cpp
namespace mlir {
#define GEN_PASS_DEF_STRIPDEBUGINFOPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
struct StripDebugInfo : public impl::StripDebugInfoPassBase<StripDebugInfo> {
  void runOnOperation() override;
};
} // namespace

void StripDebugInfo::runOnOperation() {
  auto unknownLoc = UnknownLoc::get(&getContext());
  getOperation()->walk([&](Operation *op) {
    op->setLoc(unknownLoc);
    // ... also resets block-argument locations (three nested loops) ...
  });
}
```

<sub>mlir/lib/Transforms/StripDebugInfo.cpp:14-42</sub>

And its TableGen side — this is *all* of it:

```tablegen
def StripDebugInfoPass : Pass<"strip-debuginfo"> {
  let summary = "Strip debug info from all operations";
  let description = [{
    This pass strips the IR of any location information, by replacing all
    operation locations with [`unknown`](Dialects/Builtin.md/#unknownloc).
  }];
}
```

<sub>mlir/include/mlir/Transforms/Passes.td:488-494</sub>

<!-- Speaker notes:
Let this sink in: a shipping upstream pass is 42 lines including the license header. The recipe: (1) GEN_PASS_DEF macro + include generates impl::StripDebugInfoPassBase; (2) derive from it (CRTP — the base is templated on your class); (3) override runOnOperation. That's it — flag name, description, clonePass, registration all come from TableGen.
If curious, the generated base lives in build/tools/mlir/include/mlir/Transforms/Passes.h.inc — and note clonePass — foreshadows threading: the pass manager CLONES passes to run them in parallel, which is why passes must be copy-constructible.
Naming convention: .td def name ends in "Pass" (StripDebugInfoPass) → generated base is StripDebugInfoPassBase. Old tutorials show defs without the suffix — the convention changed.
Pass<"strip-debuginfo"> with no second argument = op-agnostic pass (can be scheduled on any op). Anchored version on the exercise slide in a minute.
~2 min.
-->

---

# A pass is often just a walk + a library call ⏱

`-loop-invariant-code-motion`, complete `runOnOperation`:

```cpp
void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation()->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}
```

<sub>mlir/lib/Transforms/LoopInvariantCodeMotion.cpp:41-47</sub>

- The walk callback filters by **interface** — this pass hoists out of `scf.for`, `affine.for`, *and any future loop op* without knowing them.
- Post-order = innermost loops first — the comment relies on the default walk order you learned 10 slides ago.
- The heavy lifting (`moveLoopInvariantCode`) is a reusable library function.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: many real upstream passes are just a walk plus one library call — LICM's entire runOnOperation is five lines; read it tonight (mlir/lib/Transforms/LoopInvariantCodeMotion.cpp).
Purpose of this slide: calibrate expectations. Students assume "real passes" are thousands of lines of arcana; many are a walk plus a call into a transform library. The interface-based filter is the teaser for the later interfaces session — one sentence only: "LoopLikeOpInterface is a contract that loop-ish ops implement; walk can filter on contracts, not just concrete ops."
(+2 if presented.)
-->

---

# Your exercise starter: `-school-strength-reduce`

```tablegen
// exercises: include/School/SchoolPasses.td (description + statistics elided)
def SchoolStrengthReduce : Pass<"school-strength-reduce", "::mlir::func::FuncOp"> {
  let summary = "Rewrite muli-by-power-of-two into shli (hand-written walk)";
}
```

```cpp
// exercises: lib/School/StrengthReduce.cpp — your Exercise 1 starter (TODO text abridged)
namespace mlir::school {
#define GEN_PASS_DEF_SCHOOLSTRENGTHREDUCE
#include "School/SchoolPasses.h.inc"

namespace {
struct SchoolStrengthReduce
    : public impl::SchoolStrengthReduceBase<SchoolStrengthReduce> {
  void runOnOperation() override {
    // TODO(exercise 1, step 1): Find the candidates (walk + match).
    // TODO(exercise 1, step 2): Build the replacement ops.
    // TODO(exercise 1, step 3): Replace and erase.
  }
};
} // namespace
} // namespace mlir::school
```

The second `Pass<...>` argument **anchors** the pass: it runs on (and gets scheduled per) `func.func`. `getOperation()` is typed accordingly — it returns a `func::FuncOp`.

<!-- Speaker notes:
This is the actual shape of the starter code in the exercises/ repo — students only fill the TODO body; the .td entry, CMake, and registration in school-opt are already wired. The real .td entry additionally declares a numRewrites statistic (used by stretch goal a) and its dependent dialects.
Naming: this project's def name has no "Pass" suffix (unlike the current upstream convention mentioned on the StripDebugInfo slide) — the generated base is always <DefName>Base, hence GEN_PASS_DEF_SCHOOLSTRENGTHREDUCE and impl::SchoolStrengthReduceBase here.
Anchoring: Pass<"flag"> = op-agnostic; Pass<"flag", "::mlir::func::FuncOp"> = runs once per function. Inside an anchored pass, getOperation() returns the typed anchor op (func::FuncOp here) — no casting.
Why anchor on functions and not modules? Two reasons, both coming up: (1) parallelism — the PM runs the pass on all functions concurrently; (2) scoping — the pass may only touch IR inside its anchor.
~1 min.
-->

---

# Pass options and statistics ⏱

Declared in TableGen; materialize as *member variables* of the generated base:

```tablegen
// real upstream examples (summary/description fields elided)
def TrivialDeadCodeEliminationPass : Pass<"trivial-dce"> {
  let options = [
    Option<"recursive", "recursive", "bool", /*default=*/"true",
           "Recursively visit nested regions">,
    Option<"removeBlocks", "remove-blocks", "bool", /*default=*/"true",
           "Remove unreachable blocks">];
}
def CSEPass : Pass<"cse"> {
  let statistics = [
    Statistic<"numCSE", "num-cse'd", "Number of operations CSE'd">,
    Statistic<"numDCE", "num-dce'd", "Number of operations DCE'd">];
}
```

<sub>mlir/include/mlir/Transforms/Passes.td:103-125, 88-101</sub>

In C++: read `recursive` like a bool; assign/increment `numCSE` like an int. On the CLI (📸 captured output):

```text
$ mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(cse))" --mlir-pass-statistics
'func.func' Pipeline
  CSEPass
    (S) 2 num-cse'd - Number of operations CSE'd
    (S) 0 num-dce'd - Number of operations DCE'd
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: options and statistics are declared in the pass's TableGen entry and become plain member variables — stretch goal (a) uses the pre-declared numRewrites statistic exactly like CSE's counters.
Pre-captured — do not run live unless ahead of schedule; command: mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(cse))" --mlir-pass-statistics.
Options: Option<"cppName", "cli-arg", type, default, description> → a member you read as a plain variable inside runOnOperation, settable from the pipeline string: trivial-dce{recursive=false}.
Statistics: same idea, countable — Exercise 1 stretch goal (a) fills the pre-declared numRewrites statistic and observes it exactly like the CSE output shown (that output block is real, captured from this tree; the three '=== Pass statistics report ===' banner lines above it are trimmed).
(+2 if presented.)
-->

---

# Failing, and staying honest ⏱

**Failing:** something's broken → `signalPassFailure()`.

```cpp
if (failed(somethingImportant()))
  return signalPassFailure();   // note the return!
```

- It **sets a bit — it does not return, throw, or stop your code.** The idiom is `return signalPassFailure();`
- The pipeline stops after your pass; `PassManager::run` returns failure; the IR may be left in an invalid state.

**Staying honest:** after *every* pass, the pass manager runs the **verifier** on the result (`--verify-each`, on by default).

- Your pass must leave IR that verifies — broken dominance, wrong operand types, malformed ops get caught right after *your* pass ran, not three passes later.
- The error names the offending *op* (e.g. `error: 'test.foo' op requires attribute 'attr'`); to see which pass broke it, add `--mlir-print-ir-after-failure` or `--mlir-print-ir-after-all`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: two one-liners — signalPassFailure() only sets a bit (idiom: return signalPassFailure();), and the verifier runs after every pass, so a broken rewrite is caught right after YOUR pass ran.
signalPassFailure NOT returning is a real bug generator: students call it in an if and then fall through into code that assumes success. Make them chant the idiom.
Verifier framing: it's a safety net you'll be grateful for during the exercise — if your rewrite produces type-incorrect IR, mlir-opt tells you right after your pass runs rather than after canonicalize mangles it further. Disabling (--verify-each=false) is for performance measurements, not for making errors go away.
Live proof if asked (real, verified): mlir-opt in.mlir --pass-pipeline='builtin.module(func.func(test-pass-create-invalid-ir))' → "error: 'test.any_attr_of_i32_str' op requires attribute 'attr'" — the verifier fires between passes; the message names the op, and the IR-printing flags reveal the guilty pass.
Note the distinction: erase()-with-uses crashes IMMEDIATELY inside your pass (destructor assertion from earlier), it doesn't wait for the verifier.
(+2 if presented.)
-->

---

# 🧠 Quiz: the cursor was somewhere else ⏱

A colleague "simplifies" the worked example: *"I create the new ops at the end of the block — that spot always exists."* RAUW and erase are in the correct order!

<div class="columns">
<div>

```cpp
getOperation()->walk([&](arith::MulIOp op) {
  APInt c;
  if (!matchPattern(op.getRhs(),
                    m_ConstantInt(&c)) ||
      !c.isPowerOf2())
    return;
  OpBuilder b(
      op->getBlock()->getTerminator());
  Value shift = arith::ConstantOp::create(
      b, op.getLoc(),
      b.getIntegerAttr(op.getType(),
                       c.logBase2()));
  Value shl = arith::ShLIOp::create(
      b, op.getLoc(), op.getLhs(), shift);
  op->replaceAllUsesWith(ValueRange{shl});
  op->erase();
});
```

</div>
<div>

```mlir
func.func @f(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %r = arith.muli %x, %c8 : i32
  %s = arith.addi %r, %x : i32
  return %s : i32
}
```

The pass body runs without crashing.
What does `mlir-opt` report?

① nothing — clean output
② `operation destroyed but still has uses`
③ `operand #0 does not dominate this use`
④ the new `shli` is trivially dead and vanishes

</div>
</div>

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip together with the answers slide that follows); if skipped, say: RAUW also rewires uses ABOVE your insertion point — create replacement ops right before the op you're replacing (OpBuilder b(op)), or the after-pass verifier greets you with "operand does not dominate this use".
Give ~60-90 seconds. Most students pick ① — the RAUW/erase order is correct (deliberately granted by this quiz), so it "looks fixed". ② is reflex from the earlier spot-the-bug.
Answer: ③. The addi %s is a user of %r that sits ABOVE the terminator-anchored cursor; after RAUW it uses a value defined below itself. The pass itself runs fine — the after-pass verifier (the "Failing, and staying honest" slide: "broken dominance") rejects the result and points at an op the pass never touched.
④ is a good discussion distractor: the shli is NOT dead (%s uses it) — and nothing would delete it anyway; DCE never runs in a bare pass (Session 3).
Error text verified on this tree (minimal use-before-def repro through mlir-opt): "error: operand #0 does not dominate this use" + "note: operand defined here (op in the same block)" — emitted from the dominance check in mlir/lib/IR/Verifier.cpp.
Real upstream twin (verified in history): commit 77ba6918a14d "[mlir][linalg] Fix FoldReshapeWithGenericOpByCollapsing insertion point (#133476)" — replacement built at the consumer, an extra user sat between producer and consumer; the fix is one setInsertionPointAfter(producer) with the comment "there could be uses of `producer` between it and the `tensor.collapse_shape` op".
(+2 if presented.)
-->

---

# ✅ The verifier catches it (③) — dominance ⏱

The block after the rewrite, before verification — reading it top-down shows the bug:

```mlir
func.func @f(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %s = arith.addi %0, %x : i32      // ← now uses %0 … which is defined below!
  %c3 = arith.constant 3 : i32
  %0 = arith.shli %x, %c3 : i32     // ← created "at the end of the block"
  return %s : i32
}
```

```text
error: operand #0 does not dominate this use
note: operand defined here (op in the same block)
```

- `replaceAllUsesWith` rewires **every** use — including `%s`, **above** the cursor. Creating the ops "worked"; the SSA graph is broken anyway.
- Your pass completes normally; the **after-pass verifier** — it runs after *every* pass — rejects the IR and names an op you never touched. (That's the "broken dominance" case from *Failing, and staying honest*, made concrete.)
- **Rule: choose the insertion point from where the users are — not where it's convenient.** For a 1:1 replacement, `OpBuilder b(op)` — right before the op being replaced — dominates every existing use *by construction*. That's why the worked example does exactly that.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; paired with the quiz slide before it — skip or present the pair together.
Walk the printed IR top-down and let the room spot %0 being used before defined — the visual is the lesson.
Generalize the rule: replacements go before the old op; ops consuming a producer's result go AFTER the producer (setInsertionPointAfter). When a helper moves a borrowed builder, InsertionGuard (earlier ⏱ slide) restores the cursor.
If someone asks "why doesn't create() just refuse?": OpBuilder has no global view — inserting forward references is legal MID-mutation (you might fix them up next); only the finished IR must satisfy dominance, hence the verifier owns this check.
(+1 if presented.)
-->

---

# PassManager: a tree of pipelines ⏱

Pipelines are **anchored** and **nested**, mirroring the IR structure:

```text
PassManager  (anchor: builtin.module)
 ├─ inline                          ← runs on the module
 └─ OpPassManager (func.func)       ← nested pipeline
     ├─ canonicalize                ← runs on EVERY func.func …
     └─ cse                         ← … potentially in parallel
```

```cpp
mlir::PassManager pm(module.get()->getName());
pm.addPass(mlir::createInlinerPass());              // module-level

mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
optPM.addPass(mlir::createCanonicalizerPass());     // function-level
optPM.addPass(mlir::createCSEPass());

if (mlir::failed(pm.run(*module)))
  return 4;
```

<sub>adapted from mlir/examples/toy/Ch6/toyc.cpp:149-196</sub>

Shorthand for one-offs: `pm.addNestedPass<func::FuncOp>(createCSEPass());`

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: pipelines are anchored and nested exactly like the IR — you already saw the anchor idea on the exercise starter; in C++ the shorthand is pm.addNestedPass<func::FuncOp>(...).
Key vocabulary: ANCHOR = the op type a pipeline runs on. The tree mirrors IR nesting: module pipeline contains a func.func pipeline, just like modules contain functions.
The C++ snippet is real Toy tutorial code (condensed). Emphasize run() returns LogicalResult — failure propagates from signalPassFailure.
(+2 if presented.)
-->

---

# The nesting gotcha (you *will* hit this today) ⏱

Adding a `func.func`-anchored pass directly to a module-level pipeline is a **hard error**, not an auto-fix:

```text
$ mlir-opt in.mlir --pass-pipeline="builtin.module(affine-loop-tile)"
error: Can't add pass 'AffineLoopTiling' restricted to 'func.func' on a
PassManager intended to run on 'builtin.module', did you intend to nest?
```

…and running an anchored pipeline on the wrong op fails too:

```text
$ mlir-opt in.mlir --pass-pipeline="func.func(cse)"
error: can't run 'func.func' pass manager on 'builtin.module' op
```

**Confusingly:** the bare flag `mlir-opt -affine-loop-tile in.mlir` *works* — bare flags use **implicit nesting** and silently build `builtin.module(func.func(affine-loop-tile))`.

See what was actually built: `--dump-pass-pipeline`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: if mlir-opt greets you with "did you intend to nest?", come back to this slide — explicit --pass-pipeline strings never auto-nest, bare pass flags do.
Both error messages are real output from this tree (the "<unknown>:0:" / "in.mlir:0:0:" location prefixes and a trailing "failed to add `affine-loop-tile` with options ``" line are trimmed) — read them aloud, students will see them within minutes of starting the exercise and pattern-matching on the text saves TA round-trips.
The asymmetry: --pass-pipeline is EXPLICIT nesting (you must write the full tree, starting from the top-level anchor op or `any`); bare pass flags are IMPLICIT nesting (mlir-opt inserts the func.func level for you). Same in C++: default is Nesting::Explicit — pm.addPass(funcPass) on a module PM is a fatal error; use addNestedPass.
--dump-pass-pipeline is the truth serum: it prints the canonicalized tree including implicit nesting and all default option values. Teach it early.
(+1 if presented.)
-->

---

# Textual pipelines: the syntax ⏱

```bash
build/bin/mlir-opt in.mlir \
  --pass-pipeline="builtin.module(func.func(canonicalize{top-down=true max-iterations=5},cse))"
```

- Starts with the **top-level anchor** (`builtin.module`, or `any` for op-agnostic).
- Parentheses = nesting. **Commas separate passes.**
- Options in braces, **space-separated** `key=value` pairs — *not* commas!

```text
$ ... --dump-pass-pipeline
Pass Manager with 1 passes:
builtin.module(
  func.func(
    canonicalize{ ... max-iterations=5 ... top-down=true},
    cse
  )
)
```

(Real output, trimmed: the dump spells out *all* option values, including defaults you didn't set.)

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: in pipeline strings, commas separate passes and spaces separate options inside braces — and --dump-pass-pipeline prints the tree that was actually built.
The comma-vs-space trap: commas separate PASSES, spaces separate OPTIONS inside braces. canonicalize{top-down=true,max-iterations=5} is a parse error; canonicalize{top-down=true max-iterations=5} is right.
The dump output shown is captured from this tree with exactly the command above.
Also mention: --pass-pipeline cannot be combined with bare pass flags in the same invocation.
(+2 if presented.)
-->

---

# 🧠 Quiz: three pipelines ⏱

What does each one do — run, or error (and why)?

**A**
```bash
mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(canonicalize{top-down=true max-iterations=5},cse))"
```

**B**
```bash
mlir-opt in.mlir --pass-pipeline="func.func(cse)"
```

**C**
```bash
mlir-opt in.mlir --pass-pipeline="builtin.module(affine-loop-tile)"
```

Bonus: what does plain `mlir-opt -affine-loop-tile in.mlir` do?

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule (skip together with the answers slide that follows); if skipped, say: rule of thumb — --pass-pipeline means you spell out the whole tree, bare flags mean mlir-opt guesses it; when surprised, --dump-pass-pipeline.
~90 seconds think time. Answers (also next slide):
A: VALID. canonicalize (with those two options) then cse, on every func.func in the module; functions processed potentially in parallel.
B: ERROR — "can't run 'func.func' pass manager on 'builtin.module' op". The pipeline string must start from the top-level op of the input (builtin.module) or `any`.
C: ERROR — "Can't add pass 'AffineLoopTiling' restricted to 'func.func' ... did you intend to nest?" — affine-loop-tile is func-anchored; explicit pipelines don't auto-nest.
Bonus: WORKS — bare flags use implicit nesting, building builtin.module(func.func(affine-loop-tile)). Verify claim live with --dump-pass-pipeline if the room is skeptical.
(+2 if presented.)
-->

---

# ✅ Pipeline quiz answers ⏱

| | Verdict | Why |
|---|---|---|
| **A** | ✔ runs | Full explicit tree; `canonicalize` + `cse` on every `func.func`. |
| **B** | ✘ error | `can't run 'func.func' pass manager on 'builtin.module' op` — the string must start at the top-level anchor. |
| **C** | ✘ error | `…restricted to 'func.func' … did you intend to nest?` — explicit pipelines never auto-nest. |
| bonus | ✔ runs | Bare flags use **implicit** nesting → `builtin.module(func.func(affine-loop-tile))`. |

Rule of thumb: **`--pass-pipeline` = you spell out the whole tree. Bare flags = mlir-opt guesses the tree.** When surprised: `--dump-pass-pipeline`.

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; paired with the quiz slide before it — skip or present the pair together.
If anyone asks why explicit mode refuses to guess: pipelines are contracts; silently relocating a pass to a different anchor changes what it can see and touch (next slide: threading). The error text is deliberately educational — "did you intend to nest?".
(+1 if presented.)
-->

---

# Why anchors matter: threading ⏱

The pass manager runs an anchored pipeline **in parallel over sibling anchor ops** — all `func.func` in a module at once, one thread each (pipeline copies per thread; the whole pipeline runs per function, function by function).

From the `OperationPass` class comment — the law:

```cpp
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
```

<sub>mlir/include/mlir/Pass/Pass.h:354-360</sub>

Practical rules for your pass:

- Touch only IR **nested under `getOperation()`**. Never sibling functions, never the parent module. (Reading ancestors is permitted; mutating them is not.)
- No mutable state across `runOnOperation` calls; no global mutable state. Passes must be **copy-constructible** (remember `clonePass`?).
- Anchor ops must be **`IsolatedFromAbove`** — they can't reference SSA values from enclosing regions, so sibling-parallel passes can't race through use-def chains. That's why `func.func` carries the trait, and why passes can only be anchored on such ops (the PM enforces it at runtime).

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: the pass manager runs anchored pipelines in parallel over sibling functions, so a pass may only touch IR under its own getOperation() — and "works with --mlir-disable-threading but crashes without" means that rule was broken.
Concrete mental picture: module with 100 functions, pipeline canonicalize+cse → the PM clones the pipeline once per worker thread and parallel-for-eaches over the functions. Each function gets the ENTIRE pipeline before the thread moves on (cache-friendly), so it's NOT "canonicalize everywhere, then cse everywhere".
IsolatedFromAbove in one sentence: an op is isolated if the IR inside it cannot use values defined outside it — so two threads mutating two sibling functions can never touch the same use-list. Scheduling a pass on a non-isolated or unregistered op is a runtime error ("trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'").
If someone's pass "works with --mlir-disable-threading but crashes without" — they broke one of these rules. That flag is a diagnostic tool, not a fix.
(+2 if presented.)
-->

---

# The debugging toolkit (1): watch the IR move

| Flag | What it does |
|---|---|
| `--mlir-print-ir-before=cse` | dump IR before a specific pass |
| `--mlir-print-ir-after=cse` | dump IR after a specific pass |
| `--mlir-print-ir-before-all` / `--mlir-print-ir-after-all` | dump around *every* pass |
| `--mlir-print-ir-after-change` | after-dumps only when the pass changed something |
| `--mlir-print-ir-after-failure` | dump only when a pass fails |
| `--mlir-print-ir-tree-dir=/tmp/dumps` | one file per pass, in a directory tree |

Banner format (real):

```text
// -----// IR Dump After CSEPass: cse //----- //
```

First reflex when a pipeline misbehaves: `--mlir-print-ir-after-all` and read the story top to bottom.

<!-- Speaker notes:
Rank for beginners: after-all first (bisect by eye), then before/after=<pass> once suspicion narrows, after-change to cut noise in long pipelines, tree-dir when output gets too big for a terminal.
Fine print: --mlir-print-ir-module-scope (print the whole module instead of just the anchor op) requires --mlir-disable-threading — you can't safely print IR other threads are mutating. Nice callback to the threading slide (if presented).
~1 min.
-->

---

# The debugging toolkit (2): time, count, trace, reproduce ⏱

| Flag | What it does |
|---|---|
| `--mlir-timing` | wall time per pass (`--mlir-timing-display=list\|tree`) |
| `--mlir-pass-statistics` | print pass `Statistic`s (like CSE's `num-cse'd`) |
| `--debug-only=greedy-rewriter` | targeted `LLVM_DEBUG` tracing (`--debug` = firehose) |
| `--dump-pass-pipeline` | show the pipeline tree that was actually built |
| `--verify-each=false` | skip inter-pass verification (perf runs only!) |
| `--mlir-pass-pipeline-crash-reproducer=repro.mlir` | on failure/crash: write input IR + pipeline config into one file |

In C++ / gdb: `op->dump()` prints any op, block, region, or value to stderr.

Reproducer round-trip (📸 captured output — don't run live):

```bash
mlir-opt big.mlir --pass-pipeline="..." --mlir-pass-pipeline-crash-reproducer=repro.mlir
mlir-opt repro.mlir --run-reproducer     # replays the embedded pipeline
```

<!-- Speaker notes:
FLEX SLIDE — skip if behind schedule; if skipped, say: this table is on the cheat sheet — the one flag to remember is --mlir-pass-pipeline-crash-reproducer, which writes the input IR + pipeline into one replayable repro.mlir.
Pre-captured — do not run live unless ahead of schedule; command: mlir-opt big.mlir --pass-pipeline="..." --mlir-pass-pipeline-crash-reproducer=repro.mlir, then mlir-opt repro.mlir --run-reproducer.
Crash reproducers: the generated file is the input IR plus an embedded `mlir_reproducer` resource blob recording pipeline string, threading and verify flags; --run-reproducer replays it exactly. Add --mlir-pass-pipeline-local-reproducer (needs --mlir-disable-threading) to shrink it to just the failing pass with the IR immediately before it. This is also how you ask for help: attach repro.mlir, not a 200-line shell history.
--debug-only names to know already: pass-manager (scheduling), greedy-rewriter (Session 2's driver). Values are comma-separable.
(+2 if presented.)
-->

---

# 🔴 Live demo: `--mlir-print-ir-after-all`

```bash
build/bin/mlir-opt mul.mlir --load-pass-plugin=./mul_to_shift.so \
  --pass-pipeline='builtin.module(func.func(mul-to-shift,canonicalize))' \
  --mlir-print-ir-after-all -o /dev/null
```

```text
// -----// IR Dump After {anonymous}::MulToShiftPass: mul-to-shift //----- //
func.func @mul_by_8(%arg0: i32) -> i32 {
  %c8_i32 = arith.constant 8 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.shli %arg0, %c3_i32 : i32
  return %0 : i32
}

// -----// IR Dump After CanonicalizerPass: canonicalize{...} //----- //
func.func @mul_by_8(%arg0: i32) -> i32 {
  %c3_i32 = arith.constant 3 : i32
  %0 = arith.shli %arg0, %c3_i32 : i32
  return %0 : i32
}
```

The dead `%c8_i32` visibly survives our pass and dies in `canonicalize` — the whole story in two dumps.

<!-- Speaker notes:
Same live run as the worked-example demo — identical command with --mlir-print-ir-after-all appended; up-arrow in the same shell, nothing new to set up.
Real output (trimmed: @mul_by_7 dumps and the full canonicalize option list omitted; the banner actually reads canonicalize{cse-between-iterations=false max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}).
Things to narrate: (1) one dump per (pass, anchor-op) pair — you'll see each func dumped separately, and the func order may vary: threads! (2) the banner shows pass class name + full canonicalized pass options; (3) this flag is how you debug the exercise — school-opt supports all the same flags because they come from the PassManager, not from mlir-opt.
If time allows, re-run with --mlir-timing appended and show the per-pass timing table.
~2 min. End of lecture content — core path total ≈40 min; you should be at ≈0:45, ready for the exercise briefing.
-->

---

# Exercise 1: `-school-strength-reduce`

**Task:** in the `exercises/` project, implement the pass stub: rewrite `arith.muli %x, C` (C a power of two) → `arith.shli %x, log2(C)`. Hand-written walk — exactly today's material.

**Where:** the `// TODO(exercise 1, step N)` markers in `lib/School/StrengthReduce.cpp`; run via **`school-opt`**. Hints in `exercises/exercise1.md`.

**Checkpoints** (2–3 covered by the FileCheck test `test/exercise1/strength-reduce.mlir`; 1 verified by eye — stderr prints):

1. **Match & print** — `llvm::errs()` every `arith.muli` whose RHS is a constant power of two. *(walk, `matchPattern` + `m_ConstantInt`)*
2. **Rewrite** — build the shift-amount constant + `arith.shli` with `OpTy::create`, RAUW, erase the `muli`.
3. **Robustness** — non-constant RHS, non-power-of-two, several `muli`s per function. No crashes.

**Check your progress** (from `exercises/`):

```bash
ninja -C build && <llvm-build>/bin/llvm-lit -v build/test/exercise1
```

**Stretch goals:** (a) count rewrites in the pre-declared `numRewrites` `Statistic`, watch `--mlir-pass-statistics`; (b) handle constant on the *left* too; (c) `muli %x, 1` → just RAUW with `%x`, no new op.

<!-- Speaker notes:
Briefing ~5 min (0:45-0:50), then 30 min hands-on (0:50-1:20); instructors circulate. Solution walkthrough + wrap-up fill 1:20-1:30.
Checkpoint verification, say it precisely: checkpoints 2-3 are covered by test/exercise1/strength-reduce.mlir; checkpoint 1 has no lit test — verify it by eye from the llvm::errs() prints on stderr.
Common failure modes to watch for (from the quiz): erase before RAUW (fatal error names the op — teach them to READ it); erasing non-visited ops inside the walk (nudge to collect-then-mutate); forgetting the null-check on getDefiningOp when they try manual constant matching instead of matchPattern; getIntegerAttr with the wrong type (shift amount must have the muli's result type — the verifier catches it, point them at the error).
Checkpoint pacing: everyone should reach checkpoint 2; checkpoint 3 separates the careful from the fast; stretch (a) revisits the statistics slide; stretch (b) sets up the Session 3 debrief question "why does upstream not need the LHS case?" (canonicalization moves constants right — don't spoil it yet).
Debrief questions for the wrap-up: Who iterated until fixpoint? Who re-implemented "did anything change" bookkeeping? How annoying was the manual walk? → that pain is Session 2's motivation.
-->

---

# Recap: what you can do now

- **Model** — IR is `Operation*` all the way down: ops → regions → blocks → ops; `Value` = `OpResult` | `BlockArgument`.
- **Navigate** — parent/child accessors, `getDefiningOp` (null for block args!), `getUsers`/`getUses`.
- **Traverse** — `walk()` with typed callbacks, `WalkResult`, post-order default; collect-then-mutate.
- **Create** — `OpBuilder` insertion points (`OpBuilder(op)` = *before* `op`), `OpTy::create(b, loc, ...)`, reuse locations.
- **Rewire** — `replaceAllUsesWith` *then* `erase()`. Always in that order.
- **Package** — `Passes.td` → `GEN_PASS_DEF` → `impl::...Base` → `runOnOperation()`; options & statistics for free.
- **Run** — `--pass-pipeline="builtin.module(func.func(...))"`, nesting rules, parallel-over-functions, and the debug flags.

<!-- Speaker notes:
Run through fast — each bullet should now feel earned rather than new. Ask for a show of hands per bullet: "comfortable / shaky?" Shaky bullets tell you what to revisit in the next session's warm-up quiz.
~2 min — part of the 1:20-1:30 solution-walkthrough + wrap-up block.
-->

---

# Cheat sheet: today's API surface

```cpp
// Navigate                                  // Traverse
op->getParentOp();  op->getBlock();          root->walk([](arith::MulIOp m) { ... });
op->getOperand(0);  op->getResults();        root->walk([](Operation *o) {
v.getDefiningOp<arith::ConstantOp>();          return WalkResult::interrupt(); });
v.getUsers();  v.use_empty();  v.hasOneUse();

// Match constants                           // Create
APInt c;                                     OpBuilder b(op);        // BEFORE op!
matchPattern(v, m_ConstantInt(&c));          OpBuilder::InsertionGuard g(b);
c.isPowerOf2(); c.logBase2();                auto s = arith::ShLIOp::create(b, loc, x, y);

// Rewire (in this order!)                   // Pass
oldOp->replaceAllUsesWith(newOp->getResults());  struct P : impl::PPassBase<P> {
oldOp->erase();                                    void runOnOperation() override; };
                                                 return signalPassFailure();
```

```bash
mlir-opt in.mlir --pass-pipeline="builtin.module(func.func(my-pass{opt=1},cse))" \
  --dump-pass-pipeline --mlir-print-ir-after-all --mlir-timing --mlir-pass-statistics
```

<!-- Speaker notes:
This slide is designed to be the one screenshot students keep open during the exercise — point students at it at the end of the briefing, then return to it during wrap-up. Point at the two "in this order!" lines one last time.
~1 min (wrap-up block).
-->

---

# Further reading (in your checkout)

- **`mlir/docs/Tutorials/UnderstandingTheIRStructure.md`** — today's traversal/def-use content, with diagrams; maps 1:1 to `-test-print-nesting` / `-test-print-defuse`.
  ⚠ Its walk examples call `getFunction()` — that accessor is gone; read it as `getOperation()`.
- **`mlir/docs/PassManagement.md`** — the authoritative pass-infrastructure reference (anchoring, threading contract, options, statistics, reproducers).
  ⚠ Some console-output examples predate current behavior (old dump banners, unanchored `-pass-pipeline` strings) — trust the binary over the prose.
- **`mlir/docs/Tutorials/MlirOpt.md`** — a guided tour of `mlir-opt` flags with runnable examples.
- Source worth reading whole: `mlir/lib/Transforms/StripDebugInfo.cpp`, `LoopInvariantCodeMotion.cpp`, `mlir/test/lib/IR/TestPrintNesting.cpp`.

<!-- Speaker notes:
The staleness warnings are deliberate: teaching students that docs drift and binaries don't is itself a lesson. "When docs and mlir-opt --help disagree, the binary wins."
~1 min (wrap-up block).
-->

---

# Next session: stop writing the boring parts

Today you hand-wrote: the traversal, the match, the RAUW, the erase — and left a dead constant behind.

```cpp
// Session 2 preview: the SAME rewrite as a pattern (Exercise 2A's MulByPow2ToShl)
struct MulByPow2ToShl : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    // ... just the match + the replacement. Nothing else.
  }
};
```

A **driver** (`applyPatternsGreedily`) brings: the worklist, re-visiting changed ops, fixpoint iteration, folding, **and dead-code cleanup** — your leftover constant disappears for free.

**Session 2: Rewrite Patterns & Dialect Conversion.** Bring your Exercise 1 solution — we'll port it in ten lines.

<!-- Speaker notes:
End on the hook: everything that was fiddly today becomes declarative next time. The dead-constant cliffhanger from the demo resolves in Session 2 (greedy driver DCE) and fully in Session 3 (why folding/canonicalization exist as op hooks).
Logistics: remind students to keep their exercises/ build directory — Sessions 2 and 3 build on the same project (Exercise 2 ports this rewrite; Exercise 3 improves the school dialect itself).
~1 min (wrap-up block; session ends at 1:30). Done — thanks!
-->
