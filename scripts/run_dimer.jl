#!/usr/bin/env julia

"""
LowPrecBio -- Dimerization (2A <-> A2) with Stochastic Rounding

This script runs precision-sweep experiments for the **dimerization model**,
a nonlinear system with conservation laws: 2A <-> A2.

Uses the LowPrecBio library for models, SSA, and analysis utilities.
"""

using LowPrecBio
using Random, Statistics, StatsBase
using StochasticRounding
using Plots
using Printf

# Set default plot styling
default(; legend=:topright)
include("cli_args.jl")

# ==============================================================================
# Script-specific helpers
# ==============================================================================

"""
    conservation_summary(max_devs) -> (frac_exact, worst_dev)

Conservation summary from per-trajectory max_deviation values.
"""
function conservation_summary(max_devs)
    frac_ok = count(==(0), max_devs) / length(max_devs)
    worst   = maximum(max_devs)
    return (; frac_exact=frac_ok, worst_dev=worst)
end

# ==============================================================================
# Ensemble runner
# ==============================================================================

"""
    run_ensemble(model, t_end, n_replicas, prec, acc; mode=:mixed)

Run `n_replicas` independent SSA trajectories and collect end-time A/D counts
plus conservation diagnostics.
"""
function run_ensemble(model::DimerModel, t_end, n_replicas, prec::Symbol, acc::Symbol;
                      mode::Symbol=:mixed, seed_ssa=101, seed_sr=2026)
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)
    dual  = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual)

    As   = Vector{Int}(undef, n_replicas)
    Ds   = Vector{Int}(undef, n_replicas)
    devs = Vector{Int}(undef, n_replicas)

    @inbounds for i in 1:n_replicas
        result = ssa_dimer!(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=dual.ssa, mode=mode)
        As[i]   = result.counts_A[end]
        Ds[i]   = result.counts_D[end]
        devs[i] = result.max_deviation
    end

    return (; A=As, D=Ds, max_dev=devs)
end

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

println("="^80)
println("LowPrecBio -- Dimerization Model (2A <-> A2) with Stochastic Rounding")
println("="^80)
println()

opts = parse_cli_args(ARGS)

seed_ssa = arg_int(opts, "seed-ssa", 101)
seed_sr = arg_int(opts, "seed-sr", 2026)
t_end = arg_float(opts, "t-end", 400.0)
n_replicas = arg_int(opts, "n-replicas", 50_000)
kf = arg_float(opts, "kf", 1e-3)
kr = arg_float(opts, "kr", 0.1)
initial_A = arg_int(opts, "initial-a", 100)
initial_D = arg_int(opts, "initial-d", 0)
output_dir = arg_string(opts, "output-dir", "results")
tag = arg_string(opts, "tag", "")
skip_plots = arg_bool(opts, "skip-plots", false)
model_name = model_name_with_tag("dimer", tag)

model = DimerModel(kf=kf, kr=kr, initial_A=initial_A, initial_D=initial_D)

println("Model Parameters:")
println("-"^80)
println("  kf = $(model.kf)  [association rate]")
println("  kr = $(model.kr)  [dissociation rate]")
println("  A0 = $(model.initial_A)  [initial monomers]")
println("  D0 = $(model.initial_D)  [initial dimers]")
M0 = model.initial_A + 2 * model.initial_D
println("  M0 = A0 + 2*D0 = $M0  [conserved quantity]")
println()

println("Simulation Configuration:")
println("-"^80)
println("  Simulation time:  $t_end")
println("  Ensemble size:    $n_replicas trajectories")
println("  Seeds:            seed_ssa=$seed_ssa, seed_sr=$seed_sr")
println("  Output dir:       $output_dir")
println("  Model name:       $model_name")
println()

# Mixed precision configurations: (label, prop_precision, acc_precision)
mixed_configs = [
    ("FP64 baseline",  :fp64,     :fp64),
    ("FP32",           :fp32,     :fp32),
    ("BF16 + SR",      :bf16_sr,  :fp32),
    ("FP16 + SR",      :fp16_sr,  :fp32),
    ("BF16 RTN",       :bf16_rtn, :fp32),
    ("FP16 RTN",       :fp16_rtn, :fp32),
]

# Strict precision configurations: (label, prop_precision)
strict_configs = [
    ("STRICT BF16 + SR",  :bf16_sr),
    ("STRICT FP16 + SR",  :fp16_sr),
    ("STRICT BF16 RTN",   :bf16_rtn),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

# Run mixed precision experiments
println("Running MIXED-Precision Experiments:")
println("-"^80)

results = Dict{String, NamedTuple}()
for (i, (label, prec, acc)) in enumerate(mixed_configs)
    @printf("  [%d/%d] %-19s ", i, length(mixed_configs), label * "...")
    results[label] = run_ensemble(model, t_end, n_replicas, prec, acc;
                                  mode=:mixed, seed_ssa=seed_ssa, seed_sr=seed_sr)
    println("done")
end
println()

# Run strict precision experiments
println("Running STRICT-Precision Experiments:")
println("-"^80)

for (i, (label, prec)) in enumerate(strict_configs)
    @printf("  [%d/%d] %-22s ", i, length(strict_configs), label * "...")
    results[label] = run_ensemble(model, t_end, n_replicas, prec, :fp32;
                                  mode=:strict, seed_ssa=seed_ssa, seed_sr=seed_sr)
    println("done")
end
println()

# ==============================================================================
# RESULTS: End-time count statistics
# ==============================================================================

println("="^80)
println("RESULTS: End-Time Count Statistics (A and D)")
println("="^80)
println()

println("MIXED Precision:")
println("Precision Format    |  mean(A)  |  var(A)   |  mean(D)  |  var(D)   |     n")
println("-"^80)
for (label, _, _) in mixed_configs
    r = results[label]
    @printf("%-19s | %9.3f | %9.3f | %9.3f | %9.3f | %6d\n",
            label, mean(r.A), var(r.A), mean(r.D), var(r.D), length(r.A))
end
println()

println("STRICT Precision:")
println("Precision Format    |  mean(A)  |  var(A)   |  mean(D)  |  var(D)   |     n")
println("-"^80)
for (label, _) in strict_configs
    r = results[label]
    @printf("%-22s | %9.3f | %9.3f | %9.3f | %9.3f | %6d\n",
            label, mean(r.A), var(r.A), mean(r.D), var(r.D), length(r.A))
end
println()

# ==============================================================================
# RESULTS: Conservation law validation
# ==============================================================================

println("="^80)
println("RESULTS: Conservation Law (M = A + 2D)")
println("="^80)
println()

println("Conservation check: M = A + 2D should remain constant (= $M0)")
println()
println("MIXED Precision:")
println("Precision Format    |  frac_exact (dM=0)  |  worst_dev (dM)")
println("-"^60)
for (label, _, _) in mixed_configs
    c = conservation_summary(results[label].max_dev)
    @printf("%-19s | %18.4f | %8d\n", label, c.frac_exact, c.worst_dev)
end
println()

println("STRICT Precision:")
println("Precision Format    |  frac_exact (dM=0)  |  worst_dev (dM)")
println("-"^60)
for (label, _) in strict_configs
    c = conservation_summary(results[label].max_dev)
    @printf("%-22s | %18.4f | %8d\n", label, c.frac_exact, c.worst_dev)
end
println()

println("Interpretation:")
println("  - frac_exact = 1.0 means all trajectories preserved conservation perfectly")
println("  - worst_dev = 0 is expected (integer state updates are exact)")
println()

# ==============================================================================
# RESULTS: Wasserstein distances
# ==============================================================================

println("="^80)
println("RESULTS: Wasserstein Distance vs FP64 Baseline (Monomer A)")
println("="^80)
println()

ref = results["FP64 baseline"]

println("MIXED Precision:")
println("Precision Format    |  W1(A)  |  W1(D)")
println("-"^50)
for (label, _, _) in mixed_configs
    label == "FP64 baseline" && continue
    r = results[label]
    w_A = wasserstein_distance(ref.A, r.A)
    w_D = wasserstein_distance(ref.D, r.D)
    @printf("%-19s | %7.4f | %7.4f\n", label * " vs FP64", w_A, w_D)
end
println()

println("STRICT Precision:")
println("Precision Format    |  W1(A)  |  W1(D)")
println("-"^50)
for (label, _) in strict_configs
    r = results[label]
    w_A = wasserstein_distance(ref.A, r.A)
    w_D = wasserstein_distance(ref.D, r.D)
    @printf("%-22s | %7.4f | %7.4f\n", label * " vs FP64", w_A, w_D)
end
println()

# ==============================================================================
# Generate plots
# ==============================================================================

println("="^80)
println("Generating Plots...")
println("="^80)
println()

if skip_plots
    println("  Plot generation skipped (--skip-plots)")
else
    println("  [1/1] End-time A distribution (BF16+SR vs FP64)...")

    data1 = ref.A
    data2 = results["BF16 + SR"].A
    lo = min(minimum(data1), minimum(data2))
    hi = max(maximum(data1), maximum(data2))
    edges = collect((lo - 0.5):1:(hi + 0.5))

    h1 = fit(Histogram, data1, edges)
    h2 = fit(Histogram, data2, edges)

    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    p1 = h1.weights ./ sum(h1.weights)
    p2 = h2.weights ./ sum(h2.weights)

    p = bar(centers, p2; alpha=0.6, label="BF16 + SR", xlabel="A (monomers)", ylabel="Probability",
            title="Dimerization: end-time A distribution (BF16+SR vs FP64)")
    plot!(p, centers, p1; lw=2, label="FP64 baseline")

    savefig(p, "figures/dimer_histogram.png")
    println("      Saved: figures/dimer_histogram.png")
end

# ==============================================================================
# Save results
# ==============================================================================

println("="^80)
println("Saving Results...")
println("="^80)
println()

all_configs = vcat(
    [(l, p, a, :mixed) for (l, p, a) in mixed_configs],
    [(l, p, :fp32, :strict) for (l, p) in strict_configs],
)

# Conservation law data as extra metrics
conservation_extra = Dict{String, Any}()
for (label, _) in Iterators.flatten((mixed_configs, strict_configs))
    haskey(results, label) || continue
    c = conservation_summary(results[label].max_dev)
    conservation_extra[label] = Dict{String, Any}(
        "frac_exact" => c.frac_exact,
        "worst_dev"  => c.worst_dev,
    )
end

meta_extra = Dict{String, Any}(
    "commit_hash" => current_commit_hash(),
    "tag" => sanitize_tag(tag),
    "cli_args" => Dict(opts),
)

save_results(model_name, results;
    model=model,
    t_end=t_end,
    n_replicas=n_replicas,
    seed_ssa=seed_ssa,
    seed_sr=seed_sr,
    configs=all_configs,
    count_field=[:A, :D],
    extra_metrics=conservation_extra,
    output_dir=output_dir,
    extra_metadata=meta_extra)

println()
println("="^80)
println("Analysis Complete!")
println("="^80)
println()
