# LowPrecBio.jl

Reduced-precision arithmetic (FP16/bfloat16) with stochastic rounding for the
Stochastic Simulation Algorithm (Gillespie SSA), applied to biochemical kinetics.

Companion code for the paper:

> **Reduced-precision stochastic simulation for biochemical kinetics.**
> Tom Kimpson et al. (2026). `docs/paper/reduced_precision_mathbio.pdf`.

## Install

Requires Julia ≥ 1.10.

```bash
git clone https://github.com/tomkimpson/LowPrecBio.git
cd LowPrecBio
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Reproduce the paper

```bash
julia --project=. scripts/reproduce.jl     # all experiments + figures
julia --project=. scripts/run_telegraph.jl # individual model
```

Per-model runners: `run_birth_death.jl`, `run_schlogl.jl`, `run_telegraph.jl`,
`run_dimer.jl`, `run_repressilator.jl`. Figure scripts are `plot_*.jl`.
Raw data is regenerated into `results/` (gitignored, ~500 MB).

## Models and precision modes

| Model         | Probes                                     |
| ------------- | ------------------------------------------ |
| Birth–Death   | Mean/variance fidelity (Poisson baseline)  |
| Schlögl       | Bistable occupancy, dynamic-range limits   |
| Telegraph     | Rare events, dwell-time distributions      |
| Dimerization  | Nonlinear propensities, conservation       |
| Repressilator | Long-horizon oscillation drift             |

Precision modes: `:fp64`, `:fp32`, `:fp16_rtn`, `:fp16_sr`, `:bf16_rtn`, `:bf16_sr`
(via [`StochasticRounding.jl`](https://github.com/milankl/StochasticRounding.jl)).

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

MIT — see `LICENSE`.
