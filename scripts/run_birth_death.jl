#!/usr/bin/env julia

"""
LowPrecBio -- Birth-Death Process with Stochastic Rounding

This script runs precision-sweep experiments for the **birth-death process**,
the simplest stochastic system with an analytic steady-state distribution
(Poisson with mean birth_rate/death_rate).

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
# Poisson PMF helper (for plotting)
# ==============================================================================

poisson_pmf(k::Integer, lam::Real) = (lam^k * exp(-lam)) / factorial(big(k))

# ==============================================================================
# Ensemble runner
# ==============================================================================

"""
    run_ensemble(model, t_end, n_replicas, prec, acc; mode=:mixed)

Run `n_replicas` independent SSA trajectories and collect end-time molecule counts.
Returns a named tuple with counts vector and per-run metadata.
"""
function run_ensemble(model::BirthDeathModel, t_end, n_replicas, prec::Symbol, acc::Symbol;
                      mode::Symbol=:mixed, seed_ssa=42, seed_sr=4242)
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)
    dual  = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual)

    counts = Vector{Int}(undef, n_replicas)

    @inbounds for i in 1:n_replicas
        result = ssa_birth_death!(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=dual.ssa, mode=mode)
        counts[i] = result.counts[end]
    end

    return (; counts)
end

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

println("="^80)
println("LowPrecBio -- Birth-Death Process with Stochastic Rounding")
println("="^80)
println()

opts = parse_cli_args(ARGS)

seed_ssa = arg_int(opts, "seed-ssa", 42)
seed_sr = arg_int(opts, "seed-sr", 4242)
t_end = arg_float(opts, "t-end", 200.0)
n_replicas = arg_int(opts, "n-replicas", 50_000)
birth_rate = arg_float(opts, "birth-rate", 10.0)
death_rate = arg_float(opts, "death-rate", 0.5)
initial_population = arg_int(opts, "initial-population", 0)
output_dir = arg_string(opts, "output-dir", "results")
tag = arg_string(opts, "tag", "")
skip_plots = arg_bool(opts, "skip-plots", false)
model_name = model_name_with_tag("birth_death", tag)

# Model parameters -- steady-state mean = birth_rate/death_rate = 20 by default
model = BirthDeathModel(
    birth_rate=birth_rate,
    death_rate=death_rate,
    initial_population=initial_population,
)

println("Model Parameters:")
println("-"^80)
println("  birth_rate = $(model.birth_rate)   [birth rate]")
println("  death_rate = $(model.death_rate)   [death rate per molecule]")
println("  x0         = $(model.initial_population)  [initial molecules]")
println("  Steady-state mean (birth_rate/death_rate) = $(model.birth_rate / model.death_rate)")
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
# RESULTS: Summary statistics
# ==============================================================================

println("="^80)
println("RESULTS: End-Time Molecule Count Statistics")
println("="^80)
println()

lam_over_mu = model.birth_rate / model.death_rate
println("Theoretical Poisson steady-state: mean = variance = birth_rate/death_rate = $lam_over_mu")
println()

println("MIXED Precision:")
println("Precision Format    |    Mean    |  Variance  |     n")
println("-"^60)
for (label, _, _) in mixed_configs
    c = results[label].counts
    @printf("%-19s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
end
println()

println("STRICT Precision:")
println("Precision Format    |    Mean    |  Variance  |     n")
println("-"^60)
for (label, _) in strict_configs
    c = results[label].counts
    @printf("%-22s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
end
println()

println("Interpretation:")
println("  - For Poisson: mean = variance = birth_rate/death_rate = $lam_over_mu")
println("  - All formats should produce similar mean and variance")
println("  - Large deviations suggest precision loss or bias")
println()

# ==============================================================================
# RESULTS: Wasserstein distances
# ==============================================================================

println("="^80)
println("RESULTS: Wasserstein Distance vs FP64 Baseline")
println("="^80)
println()

println("The Wasserstein distance measures distributional similarity:")
println("  - Units: molecules (average difference)")
println("  - W1 = 0: identical distributions")
println("  - W1 < 0.1: excellent agreement")
println("  - W1 > 1.0: significant deviation")
println()

ref_counts = results["FP64 baseline"].counts

println("MIXED Precision:")
println("Precision Format    |  Wasserstein Distance")
println("-"^50)
for (label, _, _) in mixed_configs
    label == "FP64 baseline" && continue
    w = wasserstein_distance(ref_counts, results[label].counts)
    @printf("%-19s |  %18.6f\n", label * " vs FP64", w)
end
println()

println("STRICT Precision:")
println("Precision Format    |  Wasserstein Distance")
println("-"^50)
for (label, _) in strict_configs
    w = wasserstein_distance(ref_counts, results[label].counts)
    @printf("%-22s |  %18.6f\n", label * " vs FP64", w)
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
    println("  [1/1] End-time distribution vs Poisson PMF...")

    # Determine common bin range across all mixed-precision modes
    all_counts = vcat([results[l].counts for (l, _, _) in mixed_configs]...)
    lo = minimum(all_counts); hi = maximum(all_counts)
    edges = collect((lo - 0.5):1:(hi + 0.5))
    ks = lo:hi

    # Plot Poisson PMF as dashed reference line
    p = plot(ks, [Float64(poisson_pmf(k, lam_over_mu)) for k in ks];
             lw=2, ls=:dash, color=:black, label="Poisson($(round(lam_over_mu, digits=1)))")

    # Overlay each mixed-precision mode as a line histogram
    for (label, _, _) in mixed_configs
        data = results[label].counts
        h = fit(Histogram, data, edges)
        centers = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2
        probs = h.weights ./ sum(h.weights)
        plot!(p, centers, probs; lw=1.5, alpha=0.8, label=label)
    end

    xlabel!(p, "Molecule count")
    ylabel!(p, "Probability")
    title!(p, "Birth-Death: Steady-State Distribution")

    savefig(p, "figures/birth_death_histogram.png")
    println("      Saved: figures/birth_death_histogram.png")
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
    count_field=:counts,
    output_dir=output_dir,
    extra_metadata=meta_extra)

println()
println("="^80)
println("Analysis Complete!")
println("="^80)
println()
