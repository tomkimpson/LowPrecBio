#!/usr/bin/env bash
#
# Regenerate main results figures 1–4 and 6 (legend font size update).
# Figure 5 (dimer) is unchanged.
#
set -euo pipefail

JULIA=/Users/tkimpson/.julia/juliaup/julia-1.11.6+0.aarch64.apple.darwin14/bin/julia
PROJECT="--project=$(cd "$(dirname "$0")/.." && pwd)"

run() {
    echo "==> $*"
    "$JULIA" "$PROJECT" "$@"
}

# Figure 1 — birth-death
run scripts/plot_birth_death_histogram.jl
run scripts/plot_birth_death_histogram_strict.jl

# Figure 2 — Schlögl
run scripts/plot_schlogl_histogram.jl
run scripts/plot_schlogl_histogram_strict.jl
run scripts/plot_schlogl_rescaled_histogram.jl
run scripts/plot_schlogl_rescaled_histogram_strict.jl

# Figure 3 — telegraph marginals (underflow stress)
run scripts/plot_telegraph_marginals.jl
run scripts/plot_telegraph_strict_marginals.jl

# Figure 3 — telegraph marginals (bimodal, same scripts with different data)
run scripts/plot_telegraph_marginals.jl \
    --input-jld2=results/telegraph_bimodal/ensemble_data.jld2 \
    --output-stem=figures/telegraph_bimodal_marginals
run scripts/plot_telegraph_strict_marginals.jl \
    --input-jld2=results/telegraph_bimodal/ensemble_data.jld2 \
    --output-stem=figures/telegraph_bimodal_marginals_strict

# Figure 4 — telegraph dwells + phase map
run scripts/plot_telegraph_dwells_single.jl
run scripts/plot_telegraph_sweep.jl

# Figure 6 — repressilator protein distribution
run scripts/plot_repressilator_protein_dist_mixed.jl
run scripts/plot_repressilator_protein_dist_strict.jl

echo ""
echo "Done — all figures regenerated."
