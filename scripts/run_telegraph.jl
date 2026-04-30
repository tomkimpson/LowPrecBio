#!/usr/bin/env julia

"""
LowPrecBio -- Telegraph (Gene Switching) Model with Stochastic Rounding

This script runs precision-sweep experiments for the **telegraph model** of gene
expression, testing both mixed precision and strict precision modes.

Dwell times are extracted post-hoc from recorded state trajectories using the
library's `extract_dwell_times` function.

Uses the LowPrecBio library for models, SSA, and analysis utilities.
"""

using LowPrecBio
using Random, Statistics, StatsBase, Distributions
using StochasticRounding
using Printf

include("cli_args.jl")

# ==============================================================================
# Script-specific helpers
# ==============================================================================

"""
    ensure_sr_active(Tprop; seed1=1234, seed2=5678, N=20000) -> Bool

If `Tprop` is an SR type (Float16sr/BFloat16sr), verify SR randomness is engaged.
Returns `true` if SR appears active (or Tprop is not an SR type).
"""
function ensure_sr_active(Tprop; seed1=1234, seed2=5678, N=20000)
    is_sr = (Tprop === Float16sr) || (Tprop === BFloat16sr)
    if !is_sr
        return true
    end

    dual_a = DualRNG(42, seed1)
    activate_sr_rng!(dual_a)
    rng = Xoshiro(42)
    v = rand(rng, Float64, N) .* 2 .- 1
    a = Tprop.(v)

    dual_b = DualRNG(42, seed2)
    activate_sr_rng!(dual_b)
    b = Tprop.(v)

    same = all(isapprox.(Float64.(a), Float64.(b), atol=0.0))
    if same
        @error "SR appears INACTIVE for $(Tprop)"
        return false
    else
        @info "SR appears ACTIVE for $(Tprop)"
        return true
    end
end

# ==============================================================================
# Ensemble runner
# ==============================================================================

"""
    run_ensemble(model, t_end, n_replicas, prec, acc; mode=:mixed)

Run `n_replicas` independent SSA trajectories. Collects end-time counts and
extracts dwell times post-hoc from each trajectory's state vector.
"""
function run_ensemble(model::TelegraphModel, t_end, n_replicas, prec::Symbol, acc::Symbol;
                      mode::Symbol=:mixed, seed_ssa=777, seed_sr=2025)
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)
    dual  = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual)

    counts = Vector{Int}(undef, n_replicas)
    states_end = Vector{Int}(undef, n_replicas)
    all_on_dwells  = Float64[]
    all_off_dwells = Float64[]

    @inbounds for i in 1:n_replicas
        result = ssa_telegraph!(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=dual.ssa, mode=mode)
        counts[i] = result.counts[end]
        states_end[i] = result.states[end]

        # Post-hoc dwell time extraction from full trajectory
        dwells = extract_dwell_times(result.states, result.times)
        append!(all_on_dwells, dwells.on_dwells)
        append!(all_off_dwells, dwells.off_dwells)
    end

    return (; counts, states=states_end, on_dwells=all_on_dwells, off_dwells=all_off_dwells)
end

"""
    dwell_summary(res, model) -> Named tuple with rate estimates and CIs

Estimate k_on from OFF dwell times and k_off from ON dwell times using MLE.
"""
function dwell_summary(res, model)
    kon_true  = model.k_on
    koff_true = model.k_off

    # OFF dwell times -> rate of leaving OFF state = k_on
    fit_off = fit_exponential(res.off_dwells)
    # ON dwell times -> rate of leaving ON state = k_off
    fit_on  = fit_exponential(res.on_dwells)

    return (;
        k_on_true=kon_true,   k_on_hat=fit_off.rate,  k_on_CI=(fit_off.lower, fit_off.upper),  n_off=fit_off.n,
        k_off_true=koff_true, k_off_hat=fit_on.rate,   k_off_CI=(fit_on.lower, fit_on.upper),   n_on=fit_on.n
    )
end

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

println("="^80)
println("LowPrecBio -- Telegraph Model (Mixed & Strict Precision)")
println("="^80)
println()

opts = parse_cli_args(ARGS)

seed_ssa = arg_int(opts, "seed-ssa", 777)
seed_sr = arg_int(opts, "seed-sr", 2025)
t_end = arg_float(opts, "t-end", 500.0)
n_replicas = arg_int(opts, "n-replicas", 50_000)
k_on = arg_float(opts, "k-on", 0.01)
k_off = arg_float(opts, "k-off", 0.1)
alpha = arg_float(opts, "alpha", 5.0)
beta = arg_float(opts, "beta", 1.0)
initial_population = arg_int(opts, "initial-population", 0)
initial_state = arg_int(opts, "initial-state", 0)
output_dir = arg_string(opts, "output-dir", "results")
tag = arg_string(opts, "tag", "")
skip_plots = arg_bool(opts, "skip-plots", false)
model_name = model_name_with_tag("telegraph", tag)

model = TelegraphModel(
    k_on  = k_on,
    k_off = k_off,
    alpha = alpha,
    beta  = beta,
    initial_population = initial_population,
    initial_state = initial_state
)

println("Model Parameters:")
println("-"^80)
println("  k_on  = $(model.k_on)   [OFF -> ON rate]")
println("  k_off = $(model.k_off)  [ON -> OFF rate]")
println("  alpha = $(model.alpha)      [production rate]")
println("  beta  = $(model.beta)      [decay rate]")
println("  x0    = $(model.initial_population)     [initial molecules]")
println("  S0    = $(model.initial_state)     [initial state: 0=OFF]")
println()

println("Simulation Configuration:")
println("-"^80)
println("  Simulation time:  $t_end")
println("  Ensemble size:    $n_replicas trajectories")
println("  Seeds:            seed_ssa=$seed_ssa, seed_sr=$seed_sr")
println("  Output dir:       $output_dir")
println("  Model name:       $model_name")
println()

# Mixed precision configurations
mixed_configs = [
    ("FP64 baseline",  :fp64,     :fp64),
    ("FP32",           :fp32,     :fp32),
    ("BF16 + SR",      :bf16_sr,  :fp32),
    ("FP16 + SR",      :fp16_sr,  :fp32),
    ("BF16 RTN",       :bf16_rtn, :fp32),
    ("FP16 RTN",       :fp16_rtn, :fp32),
]

# Strict precision configurations
strict_configs = [
    ("STRICT BF16 + SR",  :bf16_sr),
    ("STRICT FP16 + SR",  :fp16_sr),
    ("STRICT BF16 RTN",   :bf16_rtn),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

# ==============================================================================
# RUN MIXED PRECISION EXPERIMENTS
# ==============================================================================

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

# ==============================================================================
# RUN STRICT PRECISION EXPERIMENTS
# ==============================================================================

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
# RESULTS: MIXED PRECISION
# ==============================================================================

println("="^80)
println("RESULTS: End-Time Molecule Count Statistics (MIXED)")
println("="^80)
println()

println("Precision Format    |    Mean    |  Variance  |     n")
println("-"^60)
for (label, _, _) in mixed_configs
    c = results[label].counts
    @printf("%-19s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
end
println()

# ==============================================================================
# RESULTS: STRICT PRECISION
# ==============================================================================

println("="^80)
println("RESULTS: End-Time Molecule Count Statistics (STRICT)")
println("="^80)
println()

println("Precision Format    |    Mean    |  Variance  |     n")
println("-"^60)
for (label, _) in strict_configs
    c = results[label].counts
    @printf("%-19s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
end
println()

# ==============================================================================
# RESULTS: WASSERSTEIN DISTANCES
# ==============================================================================

println("="^80)
println("RESULTS: Wasserstein Distance vs FP64")
println("="^80)
println()

ref = results["FP64 baseline"].counts

println("MIXED Precision:")
println("Precision Format    |  Wasserstein Distance")
println("-"^50)
for (label, _, _) in mixed_configs
    label == "FP64 baseline" && continue
    w = wasserstein_distance(ref, results[label].counts)
    @printf("%-19s |  %18.6f\n", label * " vs FP64", w)
end
println()

println("STRICT Precision:")
println("Precision Format    |  Wasserstein Distance")
println("-"^50)
for (label, _) in strict_configs
    w = wasserstein_distance(ref, results[label].counts)
    @printf("%-19s |  %18.6f\n", label * " vs FP64", w)
end
println()

# ==============================================================================
# RESULTS: DWELL TIME ANALYSIS
# ==============================================================================

println("="^80)
println("RESULTS: Dwell-Time Rate Estimates")
println("="^80)
println()

dw_fp64 = dwell_summary(results["FP64 baseline"], model)
dw_bfsr = dwell_summary(results["BF16 + SR"], model)
dw_fpsr = dwell_summary(results["FP16 + SR"], model)

println("Comparing estimated rates (from dwell times) vs true model parameters:")
println()
@printf("k_on  true = %.5f\n", dw_fp64.k_on_true)
@printf("  FP64:     k_on_hat = %.5f  (95%% CI: [%.5f, %.5f])\n",
        dw_fp64.k_on_hat, dw_fp64.k_on_CI[1], dw_fp64.k_on_CI[2])
@printf("  BF16+SR:  k_on_hat = %.5f  (95%% CI: [%.5f, %.5f])\n",
        dw_bfsr.k_on_hat, dw_bfsr.k_on_CI[1], dw_bfsr.k_on_CI[2])
@printf("  FP16+SR:  k_on_hat = %.5f  (95%% CI: [%.5f, %.5f])\n\n",
        dw_fpsr.k_on_hat, dw_fpsr.k_on_CI[1], dw_fpsr.k_on_CI[2])

@printf("k_off true = %.5f\n", dw_fp64.k_off_true)
@printf("  FP64:     k_off_hat = %.5f  (95%% CI: [%.5f, %.5f])\n",
        dw_fp64.k_off_hat, dw_fp64.k_off_CI[1], dw_fp64.k_off_CI[2])
@printf("  BF16+SR:  k_off_hat = %.5f  (95%% CI: [%.5f, %.5f])\n",
        dw_bfsr.k_off_hat, dw_bfsr.k_off_CI[1], dw_bfsr.k_off_CI[2])
@printf("  FP16+SR:  k_off_hat = %.5f  (95%% CI: [%.5f, %.5f])\n",
        dw_fpsr.k_off_hat, dw_fpsr.k_off_CI[1], dw_fpsr.k_off_CI[2])
println()

# ==============================================================================
# GENERATE PLOTS
# ==============================================================================

println("="^80)
println("Generating Plots...")
println("="^80)
println()

if skip_plots
    println("  Plot generation skipped (--skip-plots)")
else
    println("  Standalone plotting scripts (run separately after this script):")
    println("    julia --project=. scripts/plot_telegraph_marginals.jl")
    println("    julia --project=. scripts/plot_telegraph_strict_marginals.jl")
    println("    julia --project=. scripts/plot_telegraph_dwells.jl")
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

# Collect dwell time fit summaries as extra metrics
dwell_extra = Dict{String, Any}()
for (label, _) in Iterators.flatten((mixed_configs, strict_configs))
    haskey(results, label) || continue
    ds = dwell_summary(results[label], model)
    dwell_extra[label] = Dict{String, Any}(
        "k_on_hat"  => ds.k_on_hat,
        "k_on_CI"   => [ds.k_on_CI...],
        "k_off_hat" => ds.k_off_hat,
        "k_off_CI"  => [ds.k_off_CI...],
        "n_off"     => ds.n_off,
        "n_on"      => ds.n_on,
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
    count_field=:counts,
    extra_metrics=dwell_extra,
    output_dir=output_dir,
    extra_metadata=meta_extra)

println()
println("="^80)
println("Analysis Complete!")
println("="^80)
println()
