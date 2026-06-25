# Config And Fixtures

This directory contains lightweight tracked fixtures and small configuration-like assets.

## Current Tracked File

- `introchem_retrieval_queries.jsonl`
  Repeated retrieval sanity-check questions for the Introductory Chemistry index.

## Current Role

This directory is intentionally small.

It is used for:

- fixture-based retrieval checks
- smoke-test inputs
- future small tracked configuration assets that help reproducibility

It is not intended to become a dump for local secrets or machine-specific settings.

## Local Configuration Boundary

Keep local-only settings outside this directory or in ignored files such as:

- `.env`
- local sandbox files
- generated runtime artifacts
