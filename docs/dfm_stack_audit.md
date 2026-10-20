# DFM preprocessing and stack audit — 21 October 2026

## Evidence and stack arrangement

Inspected all open PRs (#31–#42; no other open PRs), live remote branch tips,
actual commit parents, merged #28–#30, and the current dirty research workspace.
Local `equity_data.py` is untracked on the working branch but also exists in the
saved research commit `41c437e`; `synthetic_evaluation.py` and its tests are local
untracked work. Names alone were not used to determine whether code was new.
The isolated clone protects that workspace's branch, index, and unrelated work.

#30 merged with a merge commit `a1873be` on 21 October, following #28 (`65226be`)
and #29 (`8268327`). Its original head was `15818cf`. Before this repair #31's
commit parent was `15818cf`, already reachable from main. The main and original
#30 head trees were identical. Thus #31's **actual GitHub base `main` was correct**;
its text incorrectly still named the deleted predecessor branch. The repair
rebases #31 onto actual main and replays descendants, rather than merely editing
a PR base. No merged PR was reopened or rewritten.

| PR | Head branch | Before GitHub base → after base | Original tip |
|---|---|---|---|
| #31 | `review/dfm-04-return-coordinates` | `main` → `main` | `d5c1af2` |
| #32 | `review/dfm-05-author-checkpoints` | `review/dfm-04-return-coordinates` → same | `128f37c` |
| #33 | `review/dfm-06-author-training` | `review/dfm-05-author-checkpoints` → same | `8f20d6c` |
| #34 | `review/dfm-07-frozen-ddpm` | `review/dfm-06-author-training` → same | `6e3f13d` |
| #35 | `review/dfm-08-author-sampling` | `review/dfm-07-frozen-ddpm` → same | `242bc11` |
| #36 | `review/dfm-09-ambient-ddpm-h` | `review/dfm-08-author-sampling` → same | `d32b712` |
| #37 | `review/dfm-10-ddpm-h-training` | `review/dfm-09-ambient-ddpm-h` → same | `498b7b8` |
| #38 | `review/dfm-11-ddpm-guidance` | `review/dfm-10-ddpm-h-training` → same | `b056957` |
| #39 | `review/dfm-12-ddpm-h-checkpoints` | `review/dfm-11-ddpm-guidance` → same | `854bad4` |
| #40 | `review/dfm-13-author-runner` | `review/dfm-12-ddpm-h-checkpoints` → same | `d281932` |
| #41 | `review/dfm-14-compatibility-docs` | `review/dfm-13-author-runner` → same | `4aa3084` |
| #42 | `review/dfm-15-guidance-batching` | `review/dfm-14-compatibility-docs` → same | `3ff621b` |

After: `main → #31 → #32 → #33 → #34 → #35 → #36 → #37 → #38 → #39 → #40 → #41 → #42`.
Each arrow is both the GitHub dependency and actual commit ancestry, with one
incremental commit per PR. The before structure had the same PR bases, but #31
was rooted at #30's original head instead of the current main merge tip.

## Duplication findings and decisions

| PR / functions | Existing equivalent | Behavioral differences | Decision |
|---|---|---|---|
| #31 `author_data.ReturnTransform.fit/forward/inverse`; local `equity_data.standardize_training`, `synthetic_evaluation.Standardizer` | Training mean, sample std, affine forward/inverse already existed locally | Torch float32 vs NumPy float64; covariance-derived std in synthetic; tensor buffers/device vs arrays; metadata and vector shapes | Consolidated into `return_preprocessing`; NumPy facade shares the same fit and arithmetic; compatibility import only in `author_data`. Split adapter extracted into `equity_preprocessing`; local callers updated without importing the full research work into the stack. |
| #31 `ReturnTransform.event` | `conditioning.portfolio_event_in_model_coordinates` in main | Metadata checking is additional; same nonzero-mean offset | Already delegated; retained that delegation and added order validation. No second event formula. |
| #31 factor checkpoint preprocessing checks | `ReturnTransform` parameter and asset validation | Existing loader returns NumPy float64 and preserves a tuple API | Consolidated via `NumpyStandardizer`, preserving output precision and interface. Old float-list checkpoints must not round through float32. |
| #32 `AuthorModel.checkpoint_payload/from_payload` | `reference_checkpoint.load_reference_checkpoint`, `HNetwork.save/load_checkpoint` | Pinned U-Net config/revision, image geometry, float32 network state and exclusive-create writes; different payload formats | Keep model-specific serialization. Shared preprocessing now handles common validation. A generic checkpoint framework would obscure incompatible schemas. |
| #33 `train_author_dfm` | Local equity/synthetic preparation and `DFM.train_dfm` | Released Trainer, AdamW/scheduler/EMA/drop-last versus local factor DSM optimization | Training algorithms stay separate. Added `transform=` reuse with exact asset checks, copied saved buffers, and a single float32 U-Net input cast. No unnecessary refit when comparing models. |
| #34 `FrozenAuthorReference` checks/freezing/normal/step | `reference.FrozenDFMReference`, `guided_sampling._normal` | Discrete posterior transitions and integer indexes versus OU drift/time and factor projection; same small Gaussian draw idiom | Separate references justified. Do not impose a factor-map interface on U-Net; extracting a one-line RNG draw adds no useful contract. |
| #35 `sample_author_ddpm` | `sampling.generate_samples` | DDPM posterior variance, after-transition saved endpoint, discarded tail RNG consumption versus continuous Euler increments | Separate integration loops required. Existing output and RNG parity tests retained. |
| #36 `DDPMHNetwork` MLP/probability/gradient | `h_model.HNetwork`, `wrappers.HGuidance` | Full ambient inputs and normalized discrete indexes versus reduced `G(s)x` and continuous times; both use Tanh/logsigmoid/autograd | Shared layer boilerplate is small; keep explicit coordinate assumptions. A common H interface would be a separate redesign with gradient/event binding tests. |
| #37 `train_ddpm_h`, `_loss`, `collect_ddpm_labels` | `h_model.train_h`, `terminal_label_loss`, `HNetwork.sample_trajectories` | Same Adam trajectory minibatch loop and Brier objective, but discrete uniform weights/full states/reference identity versus continuous interval weights/reduced features; gradient clearing differs | Brier loss is already shared. Recommend a separately reviewed optimizer-loop extraction with prediction/loss adapters and tests for RNG, mode restoration, finite losses and final gradients. Do not silently unify dataset or time semantics. |
| #38 `_scale`; `sampling.generate_samples`, `guided_sampling.generate_conditional_samples` | Same finite nonnegative Real validator in merged #28/local samplers | No behavioral differences | Consolidated in `checkers.validate_guidance_scale` and reused by all three entry points. Existing invalid-value and zero-scale tests retained. |
| #38 `_initial` | `guided_sampling.sample_conditional_initial` | Same log-uniform rejection core, but full DDPM initial index vs projected OU time zero; probability validation, default budgets and diagnostics differ | Recommend a focused shared rejection kernel with draw/log-probability adapters, preserving exact random consumption and public diagnostics. Keep current loops until acceptance/overshoot/error-path parity tests are established. Reuse of `evaluating` and `ConditionalSamples` already avoids other duplication. |
| #39 `save_ddpm_h/load_ddpm_h` | `h_model.HNetwork.save_checkpoint/load_checkpoint` | Same bundle concept, but exact embedded author payload/upstream revision versus factor architecture and horizons | Keep separate formats and strict validation; no safe schema-level consolidation needed. |
| #40 `run_author.main` | `scripts/run_guided_dfm.py`, local `synthetic_study` | Similar CLI/JSON/output handling; author runner handles observed arrays and DDPM commands, existing runner benchmarks H/reference rejection and local study supplies oracles | Domain orchestration stays separate. Return inverse already delegates to the shared object. CLI accepts training rows; validation/test reuse is through the saved transform/Python API. |
| #41 documentation/CI | Existing local audit/reproduction notes | Backend installation/defaults versus research audit findings | No duplicate implementation; linked shared coordinate contract and expanded CI to preprocessing, portfolio-event and factor checkpoint tests. |
| #42 batch RNG fix | Local/merged conditional batching | DDPM must interleave Gaussian starts with transitions, unlike current OU batching; intentionally matches upstream consumption | Keep dedicated fix and exact generator-state regression. Combining samplers would erase required behavior. |

Also reviewed local notebooks and source snapshots. The equity notebook's inline
inverse formulas now use the saved shared adapter; synthetic `ou_moments` stays
local and unchanged. The synthetic source manifest now includes the shared
module. Historical generated artifacts and their source snapshots were left
intact. These local integration edits are not published as new notebook/study PRs.

## Numerical assessment and validation

The authors' default float32 reduction and arithmetic remain bitwise equal to the
released `train.py` recipe, including casting before statistics. Float64 equity
preprocessing matches its previous NumPy mean/std and standardized arrays exactly
in regression tests. Synthetic `sqrt(diag(cov))` now uses direct sample std:
a seeded 4,096×64 float64 comparison measured maximum relative std change
`1.7763568394002505e-15` and maximum standardized absolute change
`7.993605777301127e-15`. This is a reduction-order change, not an assurance that
long stochastic training trajectories remain identical.

Executed with the installed pinned author backend, torch 2.5.1, CPU:

- #31 full suite: **102 tests passed**.
- Completed stack: **113 tests passed**, including actual upstream training,
  float64 shared-transform checkpoint reuse, original float32 preprocessing,
  inverse/event equivalence, invalid metadata/data, held-out reuse, CLI flows,
  zero-scale and multi-batch random-stream equality.
- Local research suite: **114 tests run, 113 passed, one opt-in notebook test
  skipped**. Then enabled `SYNTHETIC_NOTEBOOK_TEST=1`: **4 tests passed**,
  including fresh-kernel end-to-end notebook execution. The suite includes the
  **8 synthetic oracle tests**, including unchanged `ou_moments`.
- Verified every remaining PR contains one incremental commit. Patch IDs for
  #32, #34–#37, #39–#40, and #42 match their original patches; only #31, #33,
  #38 and #41 intentionally changed scope. All notebook outputs and metadata
  and all unrelated existing workspace files remain byte-for-byte intact.

The initial sandbox run hit OpenMP shared-memory initialization failures in two
subprocess tests; rerunning outside that restriction passed. These were not
ignored test failures. CUDA-specific device coverage is conditional and was not
executed on this CPU host. Full-scale paper training was not run.

## Review and merge sequence

The repository allows merge commits, squash merges, and rebase merges. Recent
#28–#30 used merge commits, but do not assume future merges preserve commits.
Review each PR against its current base, in this exact order:

`#31 → #32 → #33 → #34 → #35 → #36 → #37 → #38 → #39 → #40 → #41 → #42`.

Before **each** merge, save the current head SHAs of that PR and all remaining
descendants. A squash or rebase merge replaces commit identity, so changing the
next PR's GitHub base alone can reintroduce all prior changes in its diff.

After merging #31, fetch `main`, retarget #32 to `main`, and verify the diff.
If #31 was squashed/rebased, replay only #32's own commits with:

```bash
# Save these BEFORE merging #31; use a clean isolated checkout.
old31=$(git rev-parse origin/review/dfm-04-return-coordinates)
old32=$(git rev-parse origin/review/dfm-05-author-checkpoints)
old33=$(git rev-parse origin/review/dfm-06-author-training)
# Likewise save old34 ... old42 before any branch is rewritten.

# Once the user has reviewed and merged #31:
git fetch origin
git switch review/dfm-05-author-checkpoints
git rebase --onto origin/main "$old31"
git switch review/dfm-06-author-training
git rebase --onto review/dfm-05-author-checkpoints "$old32"
# Continue #34 ... #42 in table order, using each OLD predecessor SHA
# as the exclusion boundary and its NEW branch tip as the destination.
# Test, inspect incremental diffs, and push all rewritten refs with leases:
git push origin --force-with-lease=refs/heads/review/dfm-05-author-checkpoints:"$old32" \
  review/dfm-05-author-checkpoints:review/dfm-05-author-checkpoints
# Use the corresponding saved head for every other push as well.
gh pr edit 32 --base main
```

For every subsequent merge, apply the same rule: #32's merge exposes #33,
#33 exposes #34, #34 exposes #35, #35 exposes #36, #36 exposes #37,
#37 exposes #38, #38 exposes #39, #39 exposes #40, #40 exposes #41,
and #41 exposes #42. The newly first unmerged PR targets `main`; every later
one continues targeting its immediate predecessor. Rebase all remaining
branches in dependency order when a predecessor was rewritten, preserving
saved old boundaries. Resolve conflicts within the child PR's scope, rerun
relevant checks, and inspect each `git diff new-base...child` before pushing.
Use explicit `--force-with-lease=<ref>:<saved-sha>`; if it fails, inspect the new
remote commits before deciding how to replay them. Never substitute `--force`.

With a merge commit that preserves the old head as an ancestor of main, simply
retargeting the next PR can suffice: verify `git merge-base --is-ancestor` for
the saved merged head and inspect the new three-dot diff. If main has additional
conflicting changes, rebase the next branch and propagate to descendants anyway.
GitHub may retarget after branch deletion, but verify both base and actual diff;
do not rely on that automation. After #42 there is no descendant to restack.
No PR has been merged or approved by this task.
