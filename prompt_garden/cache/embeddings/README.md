# Prompt Garden Embedding Cache

This directory is reserved for cached embedding artifacts used during answer comparison.

Expected file pattern:

- `<scope>__<embedding_model>__<content_kind>.json`

Typical uses:

- clustering near-duplicate answers
- surfacing outlier responses
- speeding up repeated review sessions

These files are rebuildable and should remain local by default.
