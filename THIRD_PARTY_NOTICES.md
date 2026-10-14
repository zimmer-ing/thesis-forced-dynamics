# Third-party code

The [MIT License](LICENSE) covers original contributions to this repository.
It does not replace the licenses or grant rights to third-party material below.
Existing source attributions are retained in the corresponding files.

| Component | Files or sections | License text |
| --- | --- | --- |
| TCN, CMU Locus Lab | `src/models/TCN_EncDec.py` | [MIT](licenses/TCN-MIT.txt) |
| Mittag-Leffler function, Konrad Hinsen | `src/models/FDEint/mittag_leffler.py` | [BSD-3-Clause](licenses/Mittag-Leffler-BSD-3-Clause.txt) |
| FDEint, Bernd Zimmering | `src/models/FDEint/FixedStepFDEint_with_forcing.py` | [MIT](licenses/FDEint-MIT.txt) |
| Original FDE solver, Cecília Coelho / Neural-FDE paper repository | `src/models/FDEint/Original_FDE_Solver.py`; solver-derived loop in `Scripts/Evaluation/Supplementary/Evaluate_FDE_solver_unforced_zero_forcing_GPU.py` | [CC-BY-4.0](licenses/Neural-FDE-CC-BY-4.0.txt) |

## Sources and modifications

- **TCN:** [source and license](https://github.com/locuslab/TCN/tree/2f8c2b817050206397458dfd1f5a25ce8a32fe65).
  Adapted to the forecasting interface and project configuration.
- **Mittag-Leffler:** [source and license](https://github.com/khinsen/mittag-leffler/tree/f7cdf8f6a9348bc113786011b107d925e839aa10).
  The Python implementation is based on Roberto Garrappa's MATLAB method;
  its source attribution is retained.
- **FDEint:** [source and license](https://github.com/zimmer-ing/FDEint/tree/6b3bb11fe38f260f6d5fa05f8d3149b4b08e37c7).
  Extended to handle known forcing.
- **Original FDE solver:** [licensed source](https://github.com/zimmer-ing/Neural-FDE/blob/874f800796f7f8a3d84205776b5b7793c0af76c0/src/utils/Original_FDE_Solver.py)
  from the repository accompanying Bernd Zimmering, Cecília Coelho and Oliver
  Niggemann, [*Optimising Neural Fractional Differential Equations for
  Performance and Efficiency*](https://proceedings.mlr.press/v255/zimmering24a.html),
  PMLR 255, 2024. The original Coelho paper and repository references remain
  in the source. Solver logic is unchanged from the linked Neural-FDE file;
  license and source comments were added. The benchmark's copied solve loop
  places its initial buffer on the input device and uses imported solver helpers.
  The solver-derived portions retain CC-BY-4.0; independent benchmark code is MIT.

## Data and dependencies

Source-code licenses do not automatically cover downloaded datasets or the
dissertation. Dataset sources are listed in
[data preparation](Scripts/DataPreparation/README.md). Installed dependencies
retain their own licenses.
