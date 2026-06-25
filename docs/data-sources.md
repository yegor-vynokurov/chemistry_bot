# Data Sources

This document describes the current textbook corpus, its source, and the repository policy around redistribution and reproducibility.

Reviewed against the official source page on June 25, 2026.

## Primary Corpus

Current primary corpus:

- `Introductory Chemistry, 1st Canadian Edition`

Current local source files:

- `data/raw/introductory_chemistry/Introductory-Chemistry-1st-Canadian-Edition-1695676481.html`
- `data/raw/introductory_chemistry/Introductory-Chemistry-1st-Canadian-Edition-1695676494.xml`

The current parser and retrieval pipeline are tuned specifically for this book.

## Official Source

Official book page:

- `https://opentextbc.ca/introductorychemistry/`

The official book page exposes downloadable formats including:

- XHTML
- Pressbooks XML
- EPUB
- Digital PDF
- Print PDF

For this repository, the XHTML export is the primary source used by the current parser.

## License Summary

According to the official BCcampus / BC Open Textbooks page, the book is distributed under:

- `CC BY-NC-SA 4.0` for the adapted book as a whole, except where otherwise noted

The official page also notes that:

- much of the underlying textbook content by David W. Ball is under `CC BY-NC-SA 3.0`
- some later additions and changes by Jessie A. Key are separately identified
- some assets may carry their own attributions or exceptions

## Repository Policy

Current repository policy:

- the `data/` directory remains ignored by default
- the project prioritizes reproducibility through scripts over committing all generated artifacts
- the source textbook file is not yet tracked in Git while the repository cleanup is still in progress

## Redistribution Note

The official license text indicates that redistribution is permitted under the relevant Creative Commons terms, provided attribution and license conditions are respected.

However, for this repository we should still be careful because:

- the main book license is non-commercial (`NC`)
- the site says "except where otherwise noted"
- some embedded assets may have their own attribution requirements

Practical working conclusion for this repository:

- including the source textbook file in a non-commercial portfolio-style repository may be possible with proper attribution and license preservation
- before making the textbook source file a tracked Git asset, the attribution text and exception handling should be reviewed carefully

For now, the safer intermediate state is:

- keep `data/` ignored
- document the official source
- keep a reproducible acquisition and build path

## Recommended Attribution Reference

The official site provides an attribution model based on:

- Ball, D. and J. Key. (2014). `Introductory Chemistry - 1st Canadian Edition`. Victoria, B.C.: BCcampus.

The repository documentation should retain attribution if the source file is later committed.
