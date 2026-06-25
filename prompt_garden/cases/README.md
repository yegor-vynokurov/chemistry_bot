# Prompt Garden Case Sets

This directory is reserved for tracked case-set definitions used by Prompt Garden experiments.

Contract:

- one file should represent one named case set
- the file format should be `cases/<case_set_id>.json`
- case sets are source artifacts, not generated outputs

Examples of future contents:

- school-level theory QA cases
- safety-sensitive experiment request cases
- mixed question-answer cases for baseline-vs-challenger comparisons

Current status:

- the repository contract is frozen for this directory
- `default_chemistry_school_cases_v1.json` is now tracked as the first public Prompt Garden case set
- additional case-set files can be added later by promoting or extending cases from `src/chemistry_bot/promptops/eval.py`
