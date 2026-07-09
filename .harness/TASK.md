# Investigation contract

## Source

> # Files mentioned by the user:
>
> ## Archive.zip: /Users/chenzhang/Downloads/Archive.zip
>
> ## My request for Codex:
> make sure to update local branch to match remote, and do a deep code audit so that you understand the current code structure before start working. Then I need you help to do conduct research on how to improve the instrument resolution calibration process with IC model.
>
> First, the attached archive contains previous research documents and data, plus two notebooks that motivate the work: the existing instrument resolution file approach does not fit the experiment data well, and we are left with strange features that cannot be addressed. The previous shows that switching to the IC model plus JENDL database helps reduce the fitting error, but it does not solve it.
> Second, it seems like our calibration process is still under determined, due to degeneracy from multiple variables, espeically when we have to fit IC model parameters as well. The current solution is to use UDR as a starting point, but I suspect that there should be better solution.
> third, the IC instrument resolution was implemented in rush, despite mulitple round of bug fixing, I am not entire sure we get it right. And it does seems like even with current IC model, there are still strange issues we cannot resolve when deal with the real data. I need you to look into the physics to figrue out what kind of fixing or improvement we need to introduce into IC model in order to make it work for real data.
> last, the IC model involved steps are very slow, which is expected during the rushed development process as we are running out of time, but now I do believe we should consider speed optimizatoin, which I think should be possible given that the IC model is a nanalytical form whereas UDR is talbuated data taht are hard to optmized for speed.

## Investigation rules

- The deliverable is a validated mechanism or a complete elimination ledger, not disappearance of a residual feature. No data may be masked, excluded, smoothed, reweighted, or windowed merely because doing so improves the fit.
- Every exclusion must have independent justification and must be reported with results both with and without the exclusion.
- Evidence is recorded as it is gathered in `investigation/elimination-ledger.md`, including the exact command and the output artifact.
- If the available artifacts cannot validate a unique cause, the sanctioned outcome is a complete hypothesis-by-evidence elimination ledger plus the single most informative next test.

## Requirements

- [ ] R1 — “make sure to update local branch to match remote” — check: `git rev-parse main origin/main` prints the same SHA and `git rev-list --left-right --count main...origin/main` prints `0 0` before investigation work branches from it.
- [ ] R2 — “do a deep code audit so that you understand the current code structure before start working” — check: `test -s investigation/code-audit.md`; the audit traces the IC/UDR calibration entry points, parameter/data flow, numerical algorithms, tests, bindings, and known recent fixes with file-and-line evidence, and its timestamp precedes modeling recommendations.
- [ ] R3 — “conduct research on how to improve the instrument resolution calibration process with IC model” — check: `test -s investigation/report.md`; every recommended change has a code/data/literature evidence chain and a falsifiable validation test.
- [ ] R4 — “the attached archive contains previous research documents and data, plus two notebooks that motivate the work” — check: `test -s investigation/archive-inventory.md`; every archive member is inventoried by path, type, size, and role, and both notebooks plus their inputs/outputs are inspected.
- [ ] R5 reproduce — “the existing instrument resolution file approach does not fit the experiment data well, and we are left with strange features that cannot be addressed” and “switching to the IC model plus JENDL database helps reduce the fitting error, but it does not solve it” — check: exact reproduction commands, metrics, residual locations/magnitudes, and output artifacts are recorded in `investigation/elimination-ledger.md`; if an environment or missing-input blocker prevents execution, the ledger names it and records all static notebook evidence without presenting that as reproduction.
- [ ] R6 — “our calibration process is still under determined, due to degeneracy from multiple variables, espeically when we have to fit IC model parameters as well” and “The current solution is to use UDR as a starting point, but I suspect that there should be better solution.” — check: identifiability is tested with at least the Jacobian/Fisher or profile-likelihood/correlation evidence supported by the artifacts; `investigation/report.md` compares UDR initialization with source-backed alternatives and specifies a discriminating experiment.
- [ ] R7 — “the IC instrument resolution was implemented in rush, despite mulitple round of bug fixing, I am not entire sure we get it right” and “look into the physics to figrue out what kind of fixing or improvement we need to introduce into IC model in order to make it work for real data” — check: implementation equations, units, normalization, support/orientation, energy dependence, parameter bounds, convolution, and calibration objective are checked against primary sources and targeted repository tests; each suspected defect is either reproduced or explicitly left unverified.
- [ ] R8 — “the IC model involved steps are very slow” and “we should consider speed optimizatoin” — check: measured timing/profile evidence identifies dominant costs; proposed optimizations preserve the model numerically and include benchmark/parity acceptance thresholds. No speed claim may rest only on “the IC model is a nanalytical form whereas UDR is talbuated data”.
- [ ] R9 candidate ledger — every plausible cause class (reference nuclear data, IC physics/model discrepancy, calibration identifiability, energy/time calibration, data reduction/background, numerical implementation, optimizer, and performance-induced approximation) has a discriminating test — check: `test -s investigation/elimination-ledger.md` and every class has a row.
- [ ] R10 discrimination — every test actually run records the exact command, exit status, output path, observation, and what it rules in or out — check: ledger rows and preserved outputs under `investigation/evidence/`.
- [ ] R11 outcome, exactly one arm — EITHER one or more causes are validated because their mechanisms predict/reproduce the observed features without excluding affected data; OR the problem remains unresolved and the elimination ledger is complete with the most informative next test — check: `investigation/report.md` labels the achieved arm and cites the ledger evidence.
- [ ] R12 disclosure — the report lists every data exclusion with independent justification and a with/without comparison, or states “no exclusions” — check: `rg -n "Data exclusions|no exclusions" investigation/report.md`.

## Coverage mapping

- The first source sentence maps to R1–R3.
- The two “First” source sentences map to R4–R5.
- The two “Second” source sentences map to R6.
- The three “third” source sentences map to R7 and the validated-mechanism requirement R11.
- The “last” source sentence maps to R8.
- R9–R12 are mandatory investigation-integrity requirements added by the investigation template; they do not reduce any source requirement.
