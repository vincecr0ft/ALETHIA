# ALETHIA manuscript

Draft of a publication-quality writeup of the ALETHIA closed-form
attention foundation model for SMEFT inference in Drell-Yan.

The structure mirrors Chen, Glioti, Marzocca, Nardini and Wulzer,
*Unbinned multivariate observables for global SMEFT analyses from
machine learning* ([arXiv:2211.02058](https://arxiv.org/abs/2211.02058)),
with framing devices from OmniLearn
([arXiv:2404.16091](https://arxiv.org/abs/2404.16091)) and the
calibration discipline of Araz and Spannowsky
([arXiv:2512.17048](https://arxiv.org/abs/2512.17048)).

## Build

```
make
```

Requires `pdflatex`, `bibtex`, and the natbib package (standard in any
recent TeX Live).

Output: `alethia.pdf`, currently 17 pages with eight figures and one
results table.

## Figures

All eight figures in `figures/` are mirrored from
`docs/research/plots/` and `docs/research/plots/full-chain/` and are
the same PNGs that ship in the project README.

## Notes for the next iteration

Three short followups are flagged but not done:

1. The headline benchmark uses the polynomial-toy oracle; the analytic
   Drell-Yan calculator with parton distribution functions is wired but
   not yet meta-trained on. Section 4 says so explicitly, and Section 7
   carries the caveat into the discussion. A repeat of Table 1 against
   the analytic calculator is the cleanest next data point.

2. The architecture figure is omitted; every paper in the
   foundation-model cluster has one. A small TikZ block in Section 3
   would close the gap.

3. The conformal coverage plot uses the closed-loop trajectory rather
   than a stand-alone calibration figure of the form of Araz and
   Spannowsky Figure 9 (conditional coverage by physics variable). A
   dedicated calibration figure would strengthen Section 6.
