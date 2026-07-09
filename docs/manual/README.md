# qanneal — The Complete Manual (LaTeX)

`qanneal_manual.pdf` is the full self-contained manual for qanneal 0.6.1:
theory (Ising/QUBO, statistical mechanics, Suzuki–Trotter, parallel
tempering, CT-PIMC, the optimal adaptive J⊥ schedule — all derived from
first principles), the complete Python/C++ API reference, worked examples,
HPC deployment, and troubleshooting.

## Build

```bash
latexmk -pdf qanneal_manual.tex     # or: pdflatex qanneal_manual.tex (run twice)
```

Requires a TeX Live installation with `newpx`, `inconsolata`, `tcolorbox`,
`listings`, `pgfplots`, `booktabs`, `titlesec`.

## Layout

- `qanneal_manual.tex` — preamble, title page, and chapter includes
- `chapters/01…15` — one file per chapter, grouped into four parts
  (Foundations · Engines · Optimal Schedule · Using the Library) plus an
  appendix with the symbol glossary and one-page formula sheet.
