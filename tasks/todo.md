# Everyday Causal Inference — shared task list

Updated: 2026-10-10. Reconciled against student repository main at `420ea70`.

Read this file first when resuming work in `everyday-ci`. This is the authoritative
checklist for this repository; other local task notes are historical evidence.
Official student scripts and datasets are in this repository. Manuscript work
belongs to `everyday-causal-inference` and requires inspection of that repository's
current main before editing. This file does not inventory every manuscript task.

## How to use this list

1. Choose one current task and record its next action here.
2. Reproduce a historical concern with actual code and data before proposing a fix.
3. For code behavior changes, write and run a failing regression check before fixing.
4. Update the same item with verification evidence; move completed items to Done.
5. Obtain separate authorization for commit, merge, push, and deletion.

Keep original issue IDs stable. An unchecked item is a concern to investigate,
not a newly confirmed defect. Record rejected concerns as closed with evidence.
Add new concerns using the next unused ID. Keep detailed logs outside this file.

## Current

Issue 27 is implemented and verified on `fix/fuzzy-rd-python-plot`. The
Python probability curve preserves genuine binary outcomes and has no unsupported
shading. Merge and push remain pending; the original next task below is deferred.

- [x] Inspect the current script and preserve the baseline.
- [x] Add and run a failing probability/ribbon regression test.
- [x] Apply the two LOWESS argument changes and remove the fixed-width bands.
- [x] Pass the regression tests and compare R/Python figures on the same CSV.
- [x] Verify exact scope and record the results in Review.

Next proposed analytical task: **issue 5**, reproduce Chapter 5 IV uncertainty.
Next action: run the Chapter 5 noncompliance example on its actual data and compare
its standard error with a proper IV fit using the same specification. Record the
observed result under issue 5 before deciding whether a correction is needed.

## Pending

The original 28 concerns were recorded for source commit
`69df7a6fbc656fc3c09c00e7463ba770387621bd`. Their old numerical results and
line references below are historical; locate the current code before acting.
Five original issues are closed in Done. Issues 29–30 come from the later
CausalImpact migration report. No pending model was rerun to create this list.

### Setup

- [ ] **Issue 3.** Historical concern: **Chapter 13 falsely promises exact pinned Python versions.** Most shared requirements are unversioned, so students are not guaranteed the package versions used for the reported results. See [Chapter 13 Python:19](<../13. Falsification and robustness Stress test your results before someone else does.py#L19>).
  Next action: Compare the Chapter 13 installation comment with requirements.txt.
  Done when: The comment accurately describes declared versions without introducing a new constraints file.
  Evidence: original student-script report, issue 3 (historical run).

### Statistical inference

- [ ] **Issue 5.** Historical concern: **Chapter 5 R reports the wrong IV standard error.** It fits ordinary regression to predicted treatment values. The effect estimate remains valid here, but ordinary regression uncertainty does not account correctly for instrumental-variable estimation. The audit found SE **5.08 instead of 4.48**. See [Chapter 5 R:852](<../5. Experiments II sample size, power, and detecting real effects.R#L852>).
  Next action: Run the Chapter 5 noncompliance example on its actual data and compare its standard error with a proper IV fit using the same specification.
  Done when: An observed discrepancy or dismissal is recorded; any approved correction has a failing-then-passing regression check.
  Evidence: original student-script report, issue 5 (historical run).

- [ ] **Issue 6.** Historical concern: **Chapter 9 Python’s serial-correlation diagnostic tests the wrong null.** Differencing independent errors creates negative correlation between adjacent differences. Testing that correlation against zero can falsely diagnose serial correlation. See [Chapter 9 Python:189](<../9. Two-way fixed effects The old difference-in-differences.py#L189>).
  Next action: Run the current Chapter 9 residual diagnostic and compare its null with the appropriate differenced-error restriction.
  Done when: The diagnostic is justified or corrected using actual panel data and a regression check.
  Evidence: original student-script report, issue 6 (historical run).

- [ ] **Issue 7.** Historical concern: **Chapter 9 Python uses observation-based degrees of freedom with clustered uncertainty.** The pretrend test uses 991 denominator degrees of freedom despite clustering by 100 stores. A cluster-based reference changes the reported p-value slightly; the conclusion stays unchanged here. See [Chapter 9 Python:325](<../9. Two-way fixed effects The old difference-in-differences.py#L325>).
  Next action: Reproduce the Chapter 9 joint pretrend test and compare its reference distribution with the fitted clustering scheme.
  Done when: The chosen degrees of freedom and resulting p-value are justified and tested.
  Evidence: original student-script report, issue 7 (historical run).

- [ ] **Issue 8.** Historical concern: **Chapter 10 Python’s aggregate uncertainty and simultaneous bands are unsupported.** Its standard-error calculations omit covariance between effects sharing controls. It also reuses an event-study critical value for different groups of intervals. The displayed “uniform 95% bands” therefore lack validated simultaneous coverage. See [Chapter 10 Python:189](<../10. Staggered treatment The new difference-in-differences.py#L189>) and [line 396](<../10. Staggered treatment The new difference-in-differences.py#L396>).
  Next action: Reproduce Chapter 10 aggregates and bands; check covariance and simultaneous coverage separately.
  Done when: Both aggregate uncertainty and each claimed simultaneous band are supported by the implemented calculation and diagnostics.
  Evidence: original student-script report, issue 8 (historical run).

- [ ] **Issue 9.** Historical concern: **Chapter 12 Python’s targeting curve has an invalid uncertainty ribbon.** The curve subtracts the estimated overall effect, but its SE ignores that subtraction and the shared observations. At 100% inclusion, the contrast is exactly zero yet the ribbon remains wide. See [Chapter 12 Python:454](<../12. Heterogeneous treatment effects - Different people, different reactions.py#L454>).
  Next action: Run the Chapter 12 targeting curve and inspect the estimate and uncertainty at full inclusion.
  Done when: The endpoint and shared-observation covariance are handled consistently and checked against the actual implementation.
  Evidence: original student-script report, issue 9 (historical run).

### Chapter 11

- [ ] **Issue 13.** Historical concern: **Python computes important results without printing them.** Effect summaries, placebo results and residual-test results are assigned or discarded. Students running the script cannot inspect those results in the console. See [Chapter 11 Python:154](<../11. Time series methods Measuring impact without a control group.py#L154>).
  Status: partly resolved. `42186d8` prints native CausalImpact summaries and reports; C-ARIMA summaries and residual diagnostics still need inspection.
  Next action: Run Chapter 11 as a script and inventory visible C-ARIMA, placebo, and residual summaries.
  Done when: All intended computed results are visible; the already-published CausalImpact output change is retained.
  Evidence: original student-script report, issue 13 (historical run).

- [ ] **Issue 14.** Historical concern: **Python silently catches plotting failures.** Broad exception handlers substitute fallback figures or simply `pass`. A successful exit can therefore conceal incomplete output, although the audit found all expected image paths. See [Chapter 11 Python:253](<../11. Time series methods Measuring impact without a control group.py#L253>).
  Next action: Locate each surviving C-ARIMA plotting handler and reproduce its failure behavior.
  Done when: Any approved output change makes plotting failure observable without silently hiding incomplete results.
  Evidence: original student-script report, issue 14 (historical run).

- [ ] **Issue 29.** Historical concern: **Chapter 11 C-ARIMA residual p-values differ from the prose.** The migration report recorded a discrepancy while diagnostic conclusions stayed the same. Reproduce before changing either code or text.
  Next action: Reproduce Chapter 11 C-ARIMA residual tests and inspect the corresponding prose on current book main.
  Done when: Reported diagnostics match the verified code, data, and specification or explain the difference.
  Evidence: local `tasks/tfp-causalimpact-migration-20261010/REPORT.md` (historical run).

- [ ] **Issue 30.** Historical concern: **Chapter 11 C-ARIMA inference and cumulative uncertainty need reconciliation.** The migration report recorded normal-approximation output described as bootstrap inference and differing R/Python cumulative uncertainty. This is separate from the repaired R bootstrap-summary helper.
  Next action: Inspect and run the C-ARIMA inference paths in both official scripts, then compare interval construction with current book prose.
  Done when: The normal-approximation/bootstrap distinction and cumulative uncertainty are accurately stated and methodologically justified.
  Evidence: local `tasks/tfp-causalimpact-migration-20261010/REPORT.md` (historical run).

### Other output problems

- [ ] **Issue 15.** Historical concern: **Chapter 8 Python saves the wrong density figures.** The density function returns a Plotnine plot, but the script saves Matplotlib’s current figure. The audit found one white image and one duplicate spending plot. See [Chapter 8 Python:129](<../8. Regression discontinuity design Cutoffs as natural experiments.py#L129>).
  Next action: Execute Chapter 8 density plots in temporary output folders and inspect both saved images.
  Done when: Both exported figures correspond to their density-test plots and are visually reviewed.
  Evidence: original student-script report, issue 15 (historical run).

- [ ] **Issue 16.** Historical concern: **Chapter 8 Python hides manipulation-test p-values.** Printing the density-test object displays metadata rather than the decisive test results. Students cannot see the p-values they are asked to interpret. See [Chapter 8 Python:125](<../8. Regression discontinuity design Cutoffs as natural experiments.py#L125>).
  Next action: Run both Chapter 8 manipulation tests and inspect the console output.
  Done when: The relevant computed test statistics and p-values are visible to students.
  Evidence: original student-script report, issue 16 (historical run).

- [ ] **Issue 17.** Historical concern: **Chapter 9’s event-study caption overstates the result.** “No significant pre-trends” contradicts one individually significant lead. The joint pre-treatment test does not reject, which is a narrower statement. See [Chapter 9 Python:413](<../9. Two-way fixed effects The old difference-in-differences.py#L413>).
  Next action: Run the Chapter 9 event study and compare individual leads, the joint pretrend test, and the current caption.
  Done when: The caption states the finding supported by the actual tests.
  Evidence: original student-script report, issue 17 (historical run).

- [ ] **Issue 18.** Historical concern: **Chapter 14’s waterfall title overlaps its subtitle.** The financial arithmetic is coherent, but both headings occupy the same vertical space in the exported figure. See [Chapter 14 Python:358](<../14. Translating causal estimates into metrics for decision making.py#L358>).
  Next action: Generate the Chapter 14 waterfall and visually inspect its headings.
  Done when: The title and subtitle are legible and separate without changing the financial calculations.
  Evidence: original student-script report, issue 18 (historical run).

- [ ] **Issue 19.** Historical concern: **Chapters 10 and 11 suppress every Python warning.** Blanket filtering can conceal convergence and estimation warnings alongside harmless messages. See [Chapter 10 Python:25](<../10. Staggered treatment The new difference-in-differences.py#L25>) and [Chapter 11 Python:29](<../11. Time series methods Measuring impact without a control group.py#L29>).
  Next action: Run Chapters 10 and 11 with warnings recorded and identify which warning filters conceal relevant diagnostics.
  Done when: An approved warning policy preserves estimation and convergence diagnostics with proportionate checks.
  Evidence: original student-script report, issue 19 (historical run).

### Numerical consistency

- [ ] **Issue 20.** Historical concern: **Chapter 10 compares an estimate with differently weighted truth.** The cohort-weighted estimate is plotted against store-period-weighted truth. Matching cohort truth is **2.90**, while the displayed reference is **3.00**; weighting alone explains some apparent error. See [Chapter 10 R:1083](<../10. Staggered treatment The new difference-in-differences.R#L1083>) and [Python:467](<../10. Staggered treatment The new difference-in-differences.py#L467>).
  Next action: Compute both estimator weights and the simulated reference weights in the R and Python Chapter 10 scripts.
  Done when: Each comparison uses matching weights or clearly explains the different estimands.
  Evidence: original student-script report, issue 20 (historical run).

- [ ] **Issue 24.** Historical concern: **Chapter 7 R uses inconsistent uncertainty conventions.** The model summary uses robust covariance, while its confidence interval uses the default covariance. Both intervals round alike in this dataset, but the procedures differ. See [Chapter 7 R:256](<../7. Instrumental variables - When people don't do what they were assigned to.R#L256>) and [line 323](<../7. Instrumental variables - When people don't do what they were assigned to.R#L323>).
  Next action: Run Chapter 7 R and compare the covariance used for the model summary and confidence interval.
  Done when: Both uncertainty calculations use the intended convention and are checked on the actual data.
  Evidence: original student-script report, issue 24 (historical run).

### Presentation and file behavior

- [ ] **Issue 25.** Historical concern: **Some R label-border arguments are obsolete.** Chapter 5 has 19 `label.size` arguments and Chapter 9 has one. The audited ggplot2 version ignores them, so intended border styling is lost. Example: [Chapter 5 R:195](<../5. Experiments II sample size, power, and detecting real effects.R#L195>).
  Next action: Run the affected Chapter 5 and Chapter 9 figures with the current ggplot2 version and capture warnings.
  Done when: Any obsolete argument replacement preserves the intended figure styling and is visually checked.
  Evidence: original student-script report, issue 25 (historical run).

- [ ] **Issue 26.** Historical concern: **Chapter 5’s default PDF output renders poorly.** The audit found missing Unicode glyphs and cramped histogram bars in the default square PDF. Histogram counts remained intact; the defect concerns export format and readability.
  Next action: Reproduce Chapter 5 PDF output with the actual fonts and export settings.
  Done when: Glyphs and bars are legible in the exported artifact; counts are preserved.
  Evidence: original student-script report, issue 26 (historical run).

- [ ] **Issue 28.** Historical concern: **Some R scripts overwrite input CSVs.** Chapters 6, 10 and 12 deliberately regenerate seven datasets during execution. That can replace a student’s edits. The audit found no substantive data drift; preserving edited inputs is the unresolved behavior choice. Example: [Chapter 6 R:88](<../6. Causal assumptions Think first, regress later.R#L88>).
  Next action: Inventory the seven regenerated CSVs and compare before/after hashes in a disposable copy.
  Done when: Robson chooses whether regeneration remains intentional; any approved change preserves the book DGP and has a relevant regression check.
  Evidence: original student-script report, issue 28 (historical run).

## Blocked

These items need an environment or another repository inspected before deciding
whether any repair is needed. They are not confirmed current defects.

- [ ] **Issue 4.** Historical concern: **The retained R graphics library has a binary mismatch.** Packages built for R 4.5.2 failed under R 4.6.1. Rebuilding the isolated library resolved the audit failures; the live book environment was not repaired. This is an environment problem, rather than an analysis defect.
  Blocker: The current live book R library has not been verified.
  Next action: Identify the current book R executable and library, then reproduce the recorded graphics-library mismatch.
  Done when: The active environment is verified or the old finding is explicitly retired; no shared library is changed without authorization.
  Evidence: original student-script report, issue 4 (historical run).

- [ ] **Issue 21.** Historical concern: **The book understates Chapter 5’s high-noise power.** It says power falls below 20% at SD 50. The audit found **29.00% in R**, **33.30% in Python**, and **29.24% analytically**. The stale statement is at `everyday-causal-inference/power-analysis.qmd:781` (historical manuscript location).
  Blocker: Current manuscript main and its numerical claims have not been verified for this concern.
  Next action: Inspect current book main and reproduce the Chapter 5 high-noise power calculation in both official scripts.
  Done when: Any prose revision matches reproduced calculations and the book structure.
  Evidence: original student-script report, issue 21 (historical run).

- [ ] **Issue 22.** Historical concern: **Chapter 5’s peeking example has stale counts and rates.** Both scripts run **100** A/A tests; the R subtitle and book say **1,000**. The reported false-positive rates are **28.00% in R** and **36.00% in Python**, rather than the book’s roughly 20%. See [Chapter 5 R:1131](<../5. Experiments II sample size, power, and detecting real effects.R#L1131>) and `everyday-causal-inference/power-analysis.qmd:869` (historical manuscript location).
  Blocker: Current manuscript main and its numerical claims have not been verified for this concern.
  Next action: Inspect current book main and reproduce the peeking example counts, labels, and rates.
  Done when: Counts and claims match the actual implementation; randomness and execution conditions are recorded.
  Evidence: original student-script report, issue 22 (historical run).

- [ ] **Issue 23.** Historical concern: **The book’s noncompliance numbers predate the corrected generator.** The historical audit reported noncomplier mean **50.39**, combined treatment mean **52.92**, and assignment effect **2.76**. The book still reports **52.39**, **54.72**, and **4.56**. See `everyday-causal-inference/power-analysis.qmd:812` (historical manuscript location).
  Blocker: Current manuscript main and its numerical claims have not been verified for this concern.
  Next action: Inspect current book main and run the noncompliance generator used by both official scripts.
  Done when: The manuscript and scripts agree on the generator and reported quantities.
  Evidence: original student-script report, issue 23 (historical run).

## Done

These completions are supported by commits already in local and remote student
main and retained verification records. The tests below were not rerun during
this checklist-only task.

- [x] **Issue 1. Chapter 13 central dependencies.** Commit `b452307`.
  Verification: tests/test-install.R; local tasks/issue-1-verification/README.md records real manifest installs and full Chapter 13 runs.

- [x] **Issue 2. R installer dependency and failure reporting.** Commit `53e2a31`.
  Verification: tests/test-install.R exercises missing CRAN/GitHub/tidybayes packages and process exit.

- [x] **Issue 10. R CausalArima bootstrap summaries use their own model draws.** Commit `ce98c9b`.
  Verification: tests/test-ch11-causalarima.R covers main, placebo, auxiliary and with/without-X summaries.

- [x] **Issue 11. Legacy Python CausalImpact chronology/plotting defect.** Commit `42186d8`.
  Verification: tests/test_time_series_causalimpact.py checks chronology, finite intervals and native plot bands.

- [x] **Issue 12. R/Python CausalImpact weekly seasonality.** Commit `42186d8`.
  Verification: tests/test_time_series_causalimpact.py checks seasonal posterior samples in all three examples.

### Verified on the task branch

- [x] **Issue 27. Chapter 8 Python fuzzy RD probability plot.** Disabled residual reweighting in both LOWESS calls and removed the arbitrary fixed-width shading.
  Verification: `python -m unittest discover -s tests -p test_fuzzy_rd_plot.py -v` fails before the patch and passes all 3 tests after it. Same-data R/Python comparison: maximum curve difference 2.15 percentage points; jumps 63.10 and 61.85 percentage points, respectively.
  Evidence: `/tmp/fuzzy-rd-plot-fix-20261018/`; regenerated Python figure and side-by-side comparison are in the ignored `images/` directory. Merge and push remain pending.

### Other completed work

- [x] Create and verify the central task list and its narrow ignore-rule exception.
  Publication was explicitly authorized on 2026-10-10. To verify integration,
  run `git log -1 main -- tasks/todo.md` and confirm that commit is included in
  `origin/main` after fetching.
- [x] Setup comments, pinned CausalArima source, and root-directory guidance:
  `420ea70`. Five installer scenarios and the Chapter 11 R/Python checks passed
  before publication; local setup task notes retain the recorded commands.
- [x] CausalImpact comparison archived and its worktree/branch removed as authorized.
  Local archive: `tasks/archives/causalimpact-package-comparison-20261010/`.
- [x] Old audit and reconciliation WIPs and branches deleted without new archives,
  as explicitly requested. Their uncommitted contents were discarded.

## Checklist creation plan

- [x] Inventory surviving notes and reconcile statuses with current main history.
- [x] Consolidate all 28 original IDs and two later concerns into this checklist.
- [x] Verify task-file visibility, ignored artifacts, links, and the exact two-file scope.

## Review

The checklist visibility assertion failed before the ignore-rule change and passed
afterward. `git check-ignore` confirms `tasks/todo.md` is visible while local
reports, archives, and the retained setup worktree remain ignored. A Python
assertion check verified all 30 issue IDs occur once, all source-file links
resolve, the five completion commits are included in main and origin/main, and
all other tracked files match HEAD byte for byte. `git diff --check` passed.
Only `.gitignore` and `tasks/todo.md` changed. These creation-stage checks
preceded publication, which Robson subsequently authorized on 2026-10-10.
Statistical tests were not rerun for this documentation-only patch.

### Issue 27 review — 2026-10-18

The 3 regression tests failed on the original code and passed after the approved
patch. Both real plotting blocks were executed on the unchanged published CSV;
curve differences stayed below 3.00 percentage points and jump differences below
2.00 percentage points. Both figures were inspected for legibility and the fuzzy
uptake pattern. The R source, dataset, and Python estimation sections remain
byte-for-byte unchanged. R retains calculated confidence bands; Python omits
shading. No new dependency, manuscript edit, or unrelated analytical rerun was
needed. The repair changes the Chapter 8 Python source, its regression test, and
this checklist; generated images are ignored. `git diff --check` passed.

## Evidence and limits

Portable evidence is in Git history and the tracked `tests/` files. Local-only
sources used for reconciliation are `tasks/student-script-issues-2026-10-10.md`,
`tasks/issue-1-verification/README.md`,
`tasks/tfp-causalimpact-migration-20261010/REPORT.md`, and
`tasks/setup-instructions-20261010/tasks/todo.md`. These sources and their logs
remain ignored and will not be available in a fresh clone. Their historical
publication statements are superseded by the commit statuses in this checklist.

This list covers surviving known student-script concerns. It cannot recover the
deleted audit report or establish that all repository defects have been found.
The setup branch/worktree remains at `420ea70`; its removal is not authorized.


## Fuzzy RDD redesign — 2026-10-18

- [x] Inventory every book/code/media dependency.
- [x] Repair canonical generator test-first; pass frozen 500-seed checks.
- [x] Recompute and reconcile R/Python results and shared data.
- [x] Update affected prose, tables, figures and code; audit media and record user-deferred replacement.
- [x] Render full book; verify impact map and independent review.

Review: implementation complete in isolated task branches. The canonical generator passed 18 checks; four actual R/Python ratio/plot tests and the three existing probability-plot tests passed. Both full Chapter 8 scripts ran; 316 numerical fields agree to two decimals. All frozen 500-sample acceptance criteria passed. All 24 configured pages rendered; affected crossrefs, image paths, table and desktop/mobile views checked. The 30-recording audit found affected results only in Chapter 8 EN/PT; replacement is user-deferred and every MP3 is unchanged. Independent review is closed. Original dirty work and CSVs remain preserved; concurrent main commits touch unrelated review tooling only. No task-owned processes remain. No commit, merge, push or publication by this task.

Evidence: book `reports/chapter-reviews/fuzzy-rdd-redesign-20261018/REPORT.md`, exact manifest, numerical ledger, logs, media handoff and preservation record.
