---
name: code-review
description: Review code changes for quality, security, performance, and correctness following project-specific standards. Use when reviewing pull requests, examining git diffs, or when the user asks for a code review. This skill should be used proactively. When a PR/MR link or number is given, review THAT PR/MR's head fetched from the remote — never the local checkout. Only when no PR/MR is referenced, auto-detect the current branch and diff against main.
---

# Code Review

**First review step: run every registered machine-gate dimension on the pinned
checkout/base/head, consume every result and suggestion, then start semantic
review or independent simplify.** A subset profile, format/lint/metadata checks,
or a previous revision's receipt cannot substitute for this step.

## User-assigned reviews: three independent reviewers

For reviews explicitly assigned by the user, each code-review and separate
simplify pass uses three independent parallel reviewers: the current root Codex
session's model, GLM and DeepSeek. Pin the same base/head and full scope;
first-pass reviewers must not read each other's conclusions. Complete both passes
on the initial scope and record them separately. After review has no unresolved
valid findings, focus subsequent refactoring on the three independent simplify
legs and root verification of the fresh diff and callers. Reuse coverage of
unchanged code; do not restart six legs for equivalent cleanup or a rewritten
commit SHA. Run a scoped code-review when observable semantics, public APIs,
concurrency, ownership, tensor layouts or backend contracts change, prior evidence
becomes invalid, or the user explicitly requests another review. Stop after full
simplify coverage has no undecided valid candidates; do not add empty confirmation
rounds. Failed or partial passes are not a clean checkpoint. Verify coverage by
normalizing captured read receipts to the pinned path manifest and computing the
actual union and difference; a model's count or complete flag is not evidence.
Close only the exact gap after a capped run, using bounded lossless diff chunks
when needed. Each leg may reuse
its own already-read, hash-unchanged rules/context; a cold session cannot claim
that history. Codex verifies each candidate against actual code and callers;
agreement is not proof. GPT runs only in Codex; domestic reviewers read the
existing model configuration. Requests for CC review follow this workflow too.

The root Codex session orchestrates this user-assigned review. An assigned
reviewer executes only its own leg; this skill does not recursively spawn three
reviewers. Internal MR auto review is a separate domestic-only workflow. Preserve
its existing allowlist for every stage and do not add a Codex/GPT leg.

Official quota stays in the current root Codex session by default. Review and
simplify responsibilities do not authorize extra Codex/GPT subagents. Delegate
bulk reading, implementation, fixture adaptation and log summaries to domestic
models; the root verifies evidence and the necessary code. Reuse unchanged
coverage and continue each leg's own session instead of restarting full scans.
Keep full artifacts outside the conversation, return concise findings, and
checkpoint and compact long contexts regularly. Follow the quota/context policy
in docs/xllm/workflow/mr-autoreview.md; never trade away required coverage or
claim unexecuted validation passed to save quota.

## Own PR delivery: xllm-org / wai-bao

When preparing or updating either account's PR, use the knowledge repository's
current canonical review rules and complete this review once, followed by a
separate `simplify` pass for initial delivery. Subsequent updates in the same task
follow the clean-review incremental workflow above; a new PR or unrelated scope
needs its own initial coverage. A simplify perspective, machine-gate or another bot's
review does not count as that pass. The delivery task must implement valid code
fixes from both passes and recheck the final diff before publication; record the
actual scope, findings, fixes and final revision privately. When the user has
authorized fixing inherited upstream issues, verify and fix them too; an issue
being pre-existing or outside added lines is not a rejection reason. Place each
fix with its responsible existing commit when amendment is required.
Unexecuted or failed
passes are incomplete. Follow the knowledge repository's
docs/xllm/handbook/05-workflow-and-review/03-review-and-gate.md.

This skill's review pass stays read-only, including in auto review. The authorized
delivery task applies fixes after both passes; reviewing another author's PR
does not authorize editing or publishing on their behalf.

With push authorization, treat the actionable findings from one review round as
one repair batch. Finish all fixes and continue parallel review/simplify until
no actionable findings remain before starting wheel builds or unit tests.
Immediately before validation, fetch the live canonical GitHub xLLM-AI/xllm main;
if it advanced, rebase, resolve conflicts and review the affected scope first.
After the batch passes its required checks, commit/amend as authorized, push and
verify the remote ref and PR head promptly. Keep dependent edits together and
preserve per-commit validation for sequential cherry-picks. Blocking gates,
full-scope review, identity and privacy checks still apply; never bypass hooks
or fabricate receipts. Do not call unfinished checks passed.
Follow the incremental-push section in
docs/xllm/handbook/05-workflow-and-review/02-git-and-pr.md.

Every review and independent simplify pass must also feed confirmed new issues
back into the canonical `references/custom-code-style*.md` guides in the knowledge
repository, following its review-feedback section. Reviewers propose reusable
rules privately; the authorized delivery task verifies, deduplicates and lands
the rule update alongside the code fix. Updating a project copy or a review
report alone does not complete this feedback loop.

Before publishing any PR title, body, comment or review, check the final payload,
own commit metadata, branch, diff, attachments and linked content for private
identity and actual environment details. Keep logs, execution environments,
receipts and review-process records private; public text explains project changes
only. Secret scanning does not cover this contract. Re-read platform content
after creation or updates. Follow the canonical privacy rules in
docs/xllm/handbook/05-workflow-and-review/02-git-and-pr.md.

## Addressed PR comments: reply and resolve immediately

Read every review body, inline thread and ordinary PR comment, including
collapsed Nitpick and Additional comments sections. Paginate all collections;
querying unresolved threads alone is incomplete. Verify each concrete opinion
against current code regardless of its severity label, deduplicate repeated
issues, and record its disposition. Handle non-inline opinions without posting
PR replies; keep their outcomes in private records and do not claim thread
resolution. Reply and resolve only inline threads after verifying publication.
Report comments complete only after every opinion has a disposition.

For authorized inline PR-comment delivery, once an individual fix has passed the
affected checks, been committed as authorized and pushed, verify the remote and
PR head, then immediately reply and resolve its thread. Do not wait for other
comments, another review round, wheel builds or the full test suite. Keep those
checks running and report incomplete validation privately. This does not expand
a read-only reviewer's write permissions.

Use the designated account and check its identity before each platform write.
Replies describe only the code change and resulting behavior. Re-read the reply
and thread's `isResolved` after writing; check existing replies before retrying
to avoid duplicates. Leave unfixed, unpublished or contract-unverified issues
unresolved. Follow the PR-comment delivery section in
docs/xllm/handbook/05-workflow-and-review/03-review-and-gate.md.

## Review Comment Deduplication

Group findings with the same root cause and required fix before publishing. Explain the issue fully once; additional affected locations may use `Ditto: <link to the original comment>` instead of repeating the explanation. Read existing comments before every review and do not repost an unchanged issue after a push or retry. Keep findings with different causes, triggers, or fixes separate.

## Workflow

### Input preparation: pin the diff

**First decide which mode you are in. This is the single most common failure point of this skill — getting it wrong means you review the wrong code.**

- **Mode A — a PR/MR link or number was given.** The change lives on the **remote**, not in your working tree. Your currently checked-out branch is *irrelevant* and may coincidentally look similar (same topic, similar commit subjects) — **never derive the diff from local `HEAD` / the current checkout in this mode.** Fetch the PR/MR head ref explicitly and diff that.
- **Mode B — no PR/MR reference (review my local work).** Auto-detect from the current branch.

#### Mode A: PR / MR link or number

```bash
# This repo may NOT have a remote named "origin". Discover the real remotes.
git remote -v

# Fetch the PR/MR head by the hosting platform's ref convention into FETCH_HEAD.
# Try in order — the ref that resolves tells you the platform:
#   Coding (CODING.net) review style : refs/reviews/<N>
#   GitLab merge requests            : refs/merge-requests/<N>/head
#   GitHub pull requests             : refs/pull/<N>/head
git fetch <remote> refs/reviews/<N>                 # -> FETCH_HEAD
MR_HEAD=$(git rev-parse FETCH_HEAD)

# Base = the MR's real target, NOT your local branch. Use the merge-base with
# the branch the MR is opened against (usually main). For a single-commit MR
# this is simply the parent commit.
BASE=$(git merge-base FETCH_HEAD <remote>/main)     # adjust target if not main
git log  --oneline "$BASE..$MR_HEAD"
git diff --stat     "$BASE..$MR_HEAD"
git diff            "$BASE..$MR_HEAD"
```

**Sanity check before you review a single line:** confirm the commit author, subject line, and changed-file set actually match what the PR/MR link describes. If instead they match your current local branch, you based/fetched wrong — **stop and re-derive; do not fall back to reviewing the local branch.**

#### Mode B: local branch (no PR/MR reference)

```bash
# Resolve the canonical xLLM target by fetch URL, not by the remote name.
# In xllm-gh, origin is the contributor fork and upstream is the PR target.
CANONICAL_RE='(github\.com[:/]xLLM-AI/xllm|coding\.jd\.com[:/]xLLM_AI/xllm)(\.git)?$'
TARGET_REMOTE=""
for remote in upstream coding xLLM-AI origin $(git remote); do
  url="$(git config --get "remote.$remote.url" 2>/dev/null || true)"
  url="${url#https://gh-proxy.com/}"
  if [[ "$url" =~ $CANONICAL_RE ]]; then
    TARGET_REMOTE="$remote"
    break
  fi
done
[[ -n "$TARGET_REMOTE" ]] || {
  echo "cannot resolve canonical xLLM review target" >&2
  exit 2
}

git fetch "$TARGET_REMOTE" main --quiet
TARGET_REF="$TARGET_REMOTE/main"
TARGET_TIP=$(git rev-parse "$TARGET_REF")
CURRENT_BRANCH=$(git branch --show-current)
HEAD_SHA=$(git rev-parse HEAD)
BASE=$(git merge-base "$TARGET_TIP" "$HEAD_SHA")

printf 'target_ref=%s\ntarget_tip=%s\nmerge_base=%s\nhead=%s\n' \
  "$TARGET_REF" "$TARGET_TIP" "$BASE" "$HEAD_SHA"
git diff --stat "$BASE..HEAD"
git diff        "$BASE..HEAD"
```

If `CURRENT_BRANCH` is `main`, warn the user and ask which commits to review.

#### Mandatory base and scope guards

- Re-fetch and recompute the base at the start of **every** review turn. Never
  reuse a base SHA from an earlier review, session, IDE cache, or another clone.
- When a caller supplies a base, compare it with the live PR target/merge-base.
  Refuse a mismatch unless the user explicitly requests a historical diff.
- Record `target_ref`, `target_tip`, `merge_base`, `HEAD`, and changed-file count
  before producing findings.
- Read the complete diff in bounded chunks and inspect any truncated or
  persisted tool output before counting it as covered. A preview, file list or
  successful tool exit is not evidence that every hunk was read. Keep unread
  ranges explicit; incomplete coverage cannot support a completed review.
- Anchor each finding in the reviewed diff. Follow changed dependency edges
  to affected unchanged definitions, and identify the changed caller or
  contract that makes a callee relevant. An unrelated upstream issue
  outside the diff is out of scope unless the user authorized fixing inherited
  issues during delivery; a style violation alone does not expand a read-only
  review mandate.

### Step 1: Run all machine gates and resolve their input

After pinning the actual checkout/base/head, run the knowledge repository's
canonical machine gate with every ID from `dimensions.yaml` before this review
and the independent simplify pass. Explicitly pass the complete `--dim` list;
without it the diff runner defaults to the narrower `review-loop` profile.
Do not substitute `review-loop`, `authoring-hygiene`, a handpicked dimension list,
or `xllm-quality-gate.py` for the all-dimension run.
Initialize worktree state with `scripts/workspace/deploy-machine-gate.sh` and
select the reviewed checkout with `XLLM_ROOT`; a main-clone wrapper or another
SHA's successful lint is insufficient. Read the fresh run's
`review-machine-input.json`/`.md`, input snapshot and relevant raw reports.
Give independent reviewers the same machine evidence, not other reviewers'
conclusions. Privately disposition every candidate as fix/reject/defer with
concrete code/contract evidence. Implement valid fixes before starting semantic
review; do not ignore suggestions because they are advisory. List missing tools,
inputs, SKIP, TIMEOUT, parse failures and deferred work as unresolved coverage,
never as passed checks. A zero process exit does not prove all dimensions ran.
Re-run after code changes and bind the new head; do not label missing/empty,
SKIP or TIMEOUT checks as passed. This does not replace either semantic pass.

D52 supplies deterministic simplify candidates; D71 must record real project
Python, analyzed files and raw/scoped diagnostic counts. Read affected callers
when changed-line filtering excludes an interface warning. D40's review-loop
mode is explicitly source-only: it detects private helper/control-flow/setup
opportunities and materialization/layout risks, not proof of copy removal or
performance gains. Preserve Tensor truth semantics, lazy evaluation, stride,
kernel requirements and lifetimes before adopting any candidate. Follow the
handbook's machine-first workflow for artifacts and performance evidence.

Reconcile the result IDs against the complete registry and retain per-dimension
status and dispositions. The same unchanged base/head and input snapshot may
share one full machine run across independent reviewers and simplify; after code
changes, rerun all dimensions before reviewing the affected semantic scope.

### Step 2: Read project standards

Read [custom-code-style.md](references/custom-code-style.md) for the rules shared by all languages, plus [custom-code-style-cpp.md](references/custom-code-style-cpp.md) and [custom-code-style-python.md](references/custom-code-style-python.md) for each language you are editing or reviewing.
Explicitly audit those rules on changed C++ lines. `torch::`/`std::optional`,
integer width, parameter ownership, `explicit/final/override`, struct/class and
file-local scope are actionable project-style findings even without a runtime
bug. Do not confuse these with formatting nits or silently skip them after a
correctness pass. Verify equivalent APIs and cite the applicable rule.

For model/quantization configuration changes, explicitly audit parsing and
validation ownership under the guide's related-configuration section. Inspect
the complete affected loader/configuration functions and callers, and compare
existing entrypoints in the current target. Record the existing owner and the
concrete delegation, or why loaded tensors/runtime state require a local check.
Correct conditions and passing tests do not establish correct responsibility.

For every added production file/build target and new or changed forwarding or
loading API, keep a compact private ownership audit: path/symbol, actual
production consumers, existing owner/API considered, state or I/O owner, and
the reason to keep, merge or remove the boundary. Check the helper extraction
requirements against real callers, excluding tests and loop iterations.
Explicitly trace optional-feature disabled paths and repeated checkpoint reads.
A correctness-only review, a passing test or an unexamined new-file list cannot
close this audit. Missing ownership evidence means coverage is incomplete, not
that there are no findings. Reuse unchanged entries only when the relevant
callers/callees, configuration and backend/ownership contracts remain valid. After
a consolidation, recount call sites and re-inspect affected wrappers under §8 of
`custom-code-style.md`. Review only affected entries after a fix; do not create
repeated full scans or put the private audit in public PR text.

### Step 3: Review against the checklist

**C++ structure and simplification:**
- Independently audit custom-code-style-cpp §§6, 8 and 9 on all changed C++ files
  containing definitions or control flow, including tests and inline/header
  code; a correctness/reuse pass does not complete this audit.
  Check repeated conditions and fallback selection, unnecessary temporaries,
  repeated pure lookups, namespace reopening, and container construction/filling.
- Compare a concrete simpler implementation using the project's supported C++
  standard and STL. Preserve short-circuit evaluation, side effects, ownership,
  lifetime, linkage, allocation and construction costs; fewer lines alone do not
  establish an improvement. Do not add a helper or dense expression when direct
  code is clearer.
- Read the surrounding namespace blocks and each container's full fill loop;
  the declaration or changed hunk alone cannot establish equivalence. For known
  output sizes, distinguish sized construction plus assignment from reserved
  capacity plus direct construction, including nontrivial element types.
- For uncertain hot-path allocation, initialization or copy tradeoffs, follow
  custom-code-style-cpp §9's microbenchmark requirements before claiming a faster
  implementation. Include compiler/allocator/size boundaries and readability
  in the disposition; syntax or fewer lines alone cannot establish superiority.
- In the private record, list all inspected C++ files and candidate dispositions.
  For apparently reducible patterns retained, name the symbol and semantic or
  cost constraint; files without candidates may be grouped. A generic "style
  checked" or zero-findings verdict is insufficient. Apply the same checks in
  the independent simplify pass.

**xLLM Internal & Dependency Reuse (no reinvention):**
- Search xLLM's own implementations FIRST, across related modules, backends,
  shared utilities, base classes and registries; then search framework and
  third-party APIs. Dependency reuse alone does not complete this check.
- Follow custom-code-style §3's internal-reuse requirements (shared guide). Check both new
  helpers and hand-written call-site logic against existing xLLM abstractions;
  do not limit the audit to new function definitions or one named utility.
- Compare semantics before recommending reuse: dtype/device, shape/stride,
  undefined and empty inputs, ownership/aliasing, mutation, storage format,
  stream/event order and lifetime. Preserve intentional differences; do not
  replace an operation merely because another helper has a similar name.
- For tensor changes, apply shared-guide §3 to entire affected functions: check relevant
  existing helpers, dtype conversion, asynchronous lifetime and overload
  compatibility. Record adopted replacements and reasons for retained layouts
  or policy differences; distinguish existing API reuse from a missing shared
  primitive.
- Each reuse finding must identify the existing path and symbol, the duplicate
  changed code and a concrete equivalent replacement. A generic request to
  "reuse helpers" is not an actionable finding. These requirements also apply
  to manual review, simplify and auto review.
- Before writing any new helper, grep the full dependency surface: the repo
  itself, torch / torch_npu / CANN / ATB, AND the vendored/linked
  third-party libraries (third_party/: Mooncake, xllm_ops, xllm_atb_layers,
  torch_npu_ops, cutlass, sentencepiece, spdlog, brpc, cpprestsdk,
  etcd_cpp_apiv3, minja, xxHash, smhasher; system: absl, folly, Boost,
  protobuf, glog, gflags, leveldb). A self-rolled replacement for any of
  their facilities is a finding even when the code is correct.
- Classic reinvention smells to check: hand-rolled hashing where xxHash or
  absl::Hash exists; custom string formatting vs fmt/spdlog; re-built
  buffer/ring pools over existing abstractions; custom JSON handling over
  protobuf/cpprestsdk json; re-implemented async/RPC plumbing over brpc;
  tokenizer tricks outside sentencepiece; template metaprogramming that
  cutlass already provides.
- A new utility must beat every existing alternative on at least one stated
  axis (performance, clarity, ownership) or reuse the existing one.
- Duplicated logic: if the diff re-implements (even partially) a function
  that already exists anywhere in the dependency surface, report it with
  the existing symbol's `file:line` so the author can call it instead.
- Prefer the highest-level framework API (see custom-code-style-cpp §7); going
  lower (at::/c10::, raw HCCL/ACL) needs a stated reason.

**Correctness:**
- Logic handles edge cases and boundary conditions
- Error handling is comprehensive (no silent failures)
- Type safety maintained (no unsafe casts, proper use of `std::optional`)
- Resource lifecycle correct (RAII, no leaks, proper cleanup order)

**Architecture:**
- Clean separation of concerns, no layer violations
- Check dependency inversion, including acyclic violations: trace each new
  include/import, concrete type, inheritance, callback and target link to its
  policy/implementation owners. Cite the offending edge and reuse an existing
  contract, pass owned data or relocate the operation (custom-code-style-cpp §§2, 7).
  Keep intentional adapters and avoid added hot-path indirection.
- Changes align with existing patterns in the codebase
- No unnecessary coupling introduced
- Identify who owns each changed policy, state and helper. Scheduler decisions
  should not leak into block allocation, graph orchestration should not reach
  into one operator's private state, and generic transport should not own model
  semantics. Name the existing owner and concrete relocation/reuse.
- Trace small changes across xLLM submodules too; a common `xllm/` directory or
  a two-file diff does not establish architectural locality.
- Compare new configuration/state paths with same-category peers, including
  defaults, normalization, validation, Options/context propagation and consumers
  (shared-guide §4). Name the peer symbols and owner. Pure-function
  testability, equivalent output and an outside-diff peer are not reasons to skip
  this comparison. Audit orchestrators for accumulating capability matrices;
  keep model capabilities, topology and runtime combinations with their owners.
- Trace variant-specific cache state through factories, transfer and restoration;
  trace backend startup protocols through the platform owner rather than judging
  entrypoint size alone (shared-guide §4).

**Compile & Link Cost:**
- Trace public-header/import fan-out, template instantiations, rebuild scope,
  CMake dependency propagation and native-library loading (custom-code-style-cpp §2).
- Compare changed CMake target file lists across targets, distinguishing shared
  headers from duplicate compiled sources and checking the owning target.
- For affected native paths, check compile/link time and peak memory, binary
  sections/exports and runtime dependencies when growth is suspected, using
  comparable base/head builds.
  Include hygiene or a successful isolated compile does not prove cost parity.
  State missing evidence privately; findings must give a concrete code fix.

**Performance & Concurrency:**
- No performance regressions on hot paths
- Thread safety: proper locking, no data races
- CUDA/NPU kernels: memory coalescing, occupancy, sync correctness
- No unnecessary copies of large objects (tensors, vectors)
- Trace scheduler -> engine -> worker queues, thread-pool widths, waits and
  synchronization. Check what actually executes in parallel, regardless of an
  `async`/`no_sync` name.
- Check default/feature-disabled paths for new unconditional work. For graphs,
  distinguish captured buffer addresses from reassigned objects and trace first
  capture, replay, checkpoint, slot rotation, first decode and warm decode.

**Algorithm and configuration contracts:**
- Trace distributions through speculative acceptance/rejection/resampling and
  dtype/alias/writeback through recurrent state. A selected-token probability
  does not preserve a full proposal distribution.
- Trace new configuration through loader -> ModelArgs -> binding -> consumer;
  reading a field in one layer does not establish production reachability.
- Inspect supported eager/graph, overlap/double-buffer, TP/DP, backend and
  speculative/non-speculative combinations affected by changed conditions.
- Inspect existing tests/build/benchmark code as evidence. Report a wrong code
  operation in those files when appropriate; missing tests or measurements
  alone are not findings and must not become author-facing requests.

**Production Readiness:**
- Backward compatibility maintained (or breaking changes documented)
- Migration strategy for schema/config changes
- No hardcoded values that should be configurable

### Step 4: Output findings

Use the format below. Publish only concrete code modifications anchored in the
reviewed diff. Each issue needs the operation/rule, impact or supported trigger,
and an explicit replacement, deletion, reuse or relocation. Keep evidence limits
in the internal review record. Do not ask the author to add tests, experiments,
benchmarks, profiling, further verification, documentation or process changes;
do not append those requests to a valid code fix.

## Output Format

### Issues

#### Critical (Must Fix)

[Bugs, security holes, data loss risks, broken functionality]

#### Important (Should Fix)

[Concrete architecture changes, missing error handling, performance issues]

#### Minor (Nice to Have)

[Concrete project-style, reuse, dead-code and simplification changes]

**Each issue must include:**
- **File:line** reference
- **What** is wrong
- **Why** it matters
- **How** to fix: explicit code change, even for style/design findings

### Assessment

**Ready to merge?** [Yes / No / With fixes]

**Reasoning:** [1-2 sentence technical assessment]

## Rules

**DO:**
- Apply project-specific style from [custom-code-style.md](references/custom-code-style.md) and the matching language guide [custom-code-style-cpp.md](references/custom-code-style-cpp.md) / [custom-code-style-python.md](references/custom-code-style-python.md)
- Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable
- Categorize by actual severity (not everything is Critical)
- Be specific with file:line references
- Explain WHY issues matter
- Give a clear verdict

**DON'T:**
- Approve without thorough review
- Reuse a previously computed base SHA without fetching and recomputing it
- Mark nitpicks as Critical
- Give feedback on code not in the diff
- Be vague (e.g., "improve error handling" without specifics)
- Emit a test/experiment/documentation request or manufacture a guard merely
  because runtime evidence is absent
