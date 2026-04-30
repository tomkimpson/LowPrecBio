#!/usr/bin/env julia

"""
LowPrecBio -- Repressilator Model with Stochastic Rounding

This script runs precision-sweep experiments for the **repressilator model**, a
three-gene oscillatory circuit (A ⊣ B ⊣ C ⊣ A). Tests both mixed precision and
strict precision modes, with oscillation fidelity analysis.

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

# ------------------------------------------------------------------------------
# Script-specific helpers
# ------------------------------------------------------------------------------

"""
    bin_trajectory(times, counts, dt)

Bin an SSA trajectory into uniform time bins of width `dt`. Returns the
bin-centre times and the corresponding count value (last count before each
bin edge, i.e. a zero-order hold). This smooths out the event-level
single-step noise so that peak detection can identify oscillation cycles.
"""
function bin_trajectory(times, counts, dt)
    t_end = times[end]
    bin_edges = collect(0.0:dt:t_end)
    n_bins = length(bin_edges) - 1
    bin_times = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:n_bins]
    bin_counts = Vector{Float64}(undef, n_bins)

    j = 1
    for i in 1:n_bins
        t_right = bin_edges[i+1]
        while j < length(times) && times[j+1] <= t_right
            j += 1
        end
        bin_counts[i] = Float64(counts[j])
    end
    return bin_times, bin_counts
end

"""
    count_peaks(counts; min_prominence=5)

Detect peaks in a (binned) time series. A peak is a local maximum where
`counts[i] > counts[i-1]` and `counts[i] > counts[i+1]` with minimum
prominence (height above the mean of its neighbours) to filter noise.
Returns a vector of peak indices.
"""
function count_peaks(counts; min_prominence=5)
    peaks = Int[]
    for i in 2:(length(counts) - 1)
        if counts[i] > counts[i-1] && counts[i] > counts[i+1]
            neighbour_mean = (counts[i-1] + counts[i+1]) / 2
            if counts[i] - neighbour_mean >= min_prominence
                push!(peaks, i)
            end
        end
    end
    return peaks
end

"""
    oscillation_stats(times, protein_counts; bin_dt=1.0)

Extract oscillation metrics from a single SSA trajectory. The trajectory is
first binned into uniform intervals of width `bin_dt` to smooth event-level
noise, then peaks are detected.
Returns a named tuple `(n_peaks, mean_amplitude, mean_period)`.
"""
function oscillation_stats(times, protein_counts; bin_dt=1.0)
    bin_times, bin_counts = bin_trajectory(times, protein_counts, bin_dt)
    peak_idxs = count_peaks(bin_counts)
    n_peaks = length(peak_idxs)

    if n_peaks == 0
        return (; n_peaks=0, mean_amplitude=0.0, mean_period=NaN)
    end

    mean_amplitude = mean(bin_counts[i] for i in peak_idxs)

    if n_peaks >= 2
        peak_times = [bin_times[i] for i in peak_idxs]
        periods = diff(peak_times)
        mean_period = mean(periods)
    else
        mean_period = NaN
    end

    return (; n_peaks, mean_amplitude, mean_period)
end

# ------------------------------------------------------------------------------
# Ensemble runner
# ------------------------------------------------------------------------------

"""
    run_ensemble(model, t_end, n_replicas, prec, acc; mode=:mixed, ...)

Run `n_replicas` independent SSA trajectories. Collects per-replica end-time
counts for all 6 species and oscillation metrics (peak count, mean amplitude)
for protein A.
"""
function run_ensemble(model::RepressilatorModel, t_end, n_replicas, prec::Symbol, acc::Symbol;
                      mode::Symbol=:mixed, seed_ssa=123, seed_sr=3030,
                      max_events::Int=typemax(Int))
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)
    dual  = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual)

    end_pA = Vector{Int}(undef, n_replicas)
    peak_counts = Vector{Int}(undef, n_replicas)
    peak_amplitudes = Vector{Float64}(undef, n_replicas)

    @inbounds for i in 1:n_replicas
        result = ssa_repressilator!(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=dual.ssa,
                                    mode=mode, max_events=max_events)
        end_pA[i] = result.counts_pA[end]

        ostats = oscillation_stats(result.times, result.counts_pA)
        peak_counts[i] = ostats.n_peaks
        peak_amplitudes[i] = ostats.mean_amplitude
    end

    return (; end_pA, peak_counts, peak_amplitudes)
end

# ------------------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------------------

println("="^80)
println("LowPrecBio -- Repressilator Model (Mixed & Strict Precision)")
println("="^80)
println()

opts = parse_cli_args(ARGS)

seed_ssa = arg_int(opts, "seed-ssa", 123)
seed_sr = arg_int(opts, "seed-sr", 3030)
t_end = arg_float(opts, "t-end", 200.0)
n_replicas = arg_int(opts, "n-replicas", 1_000)
alpha0 = arg_float(opts, "alpha0", 1.0)
alpha = arg_float(opts, "alpha", 216.0)
hill_n = arg_int(opts, "hill-n", 2)
delta_m = arg_float(opts, "delta-m", 1.0)
beta = arg_float(opts, "beta", 5.0)
delta_p = arg_float(opts, "delta-p", 1.0)
output_dir = arg_string(opts, "output-dir", "results")
tag = arg_string(opts, "tag", "")
skip_plots = arg_bool(opts, "skip-plots", false)
model_name = model_name_with_tag("repressilator", tag)

model = RepressilatorModel(
    alpha0 = alpha0,
    alpha  = alpha,
    n      = hill_n,
    delta_m = delta_m,
    beta   = beta,
    delta_p = delta_p,
    initial_mA = 0,
    initial_mB = 0,
    initial_mC = 0,
    initial_pA = 5,   # asymmetric to break symmetry
    initial_pB = 0,
    initial_pC = 0,
)

println("Model Parameters:")
println("-"^80)
println("  alpha0  = $(model.alpha0)   [basal transcription rate]")
println("  alpha   = $(model.alpha) [max regulated transcription rate]")
println("  n       = $(model.n)     [Hill coefficient]")
println("  delta_m = $(model.delta_m)   [mRNA degradation rate]")
println("  beta    = $(model.beta)   [translation rate]")
println("  delta_p = $(model.delta_p)   [protein degradation rate]")
println("  Initial: mA=$(model.initial_mA), mB=$(model.initial_mB), mC=$(model.initial_mC), pA=$(model.initial_pA), pB=$(model.initial_pB), pC=$(model.initial_pC)")
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
# NOTE: Strict RTN modes (BF16 RTN, FP16 RTN) will stagnate for the repressilator.
# The high event rate (~400+ events/time unit) causes time stagnation in strict mode:
# when t grows large, τ is too small to change t under round-to-nearest, so the SSA
# loop never terminates. A max_events guard prevents infinite loops; the resulting
# trajectories will be incomplete, documenting the failure mode.
strict_configs = [
    ("STRICT BF16 + SR",  :bf16_sr),
    ("STRICT FP16 + SR",  :fp16_sr),
    ("STRICT BF16 RTN",   :bf16_rtn),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

# ------------------------------------------------------------------------------
# RUN MIXED PRECISION EXPERIMENTS
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# RUN STRICT PRECISION EXPERIMENTS
# ------------------------------------------------------------------------------

println("Running STRICT-Precision Experiments:")
println("-"^80)

# Use max_events guard for strict RTN modes to prevent infinite loops from time stagnation
max_events_strict = 500_000
for (i, (label, prec)) in enumerate(strict_configs)
    @printf("  [%d/%d] %-22s ", i, length(strict_configs), label * "...")
    results[label] = run_ensemble(model, t_end, n_replicas, prec, :fp32;
                                  mode=:strict, seed_ssa=seed_ssa, seed_sr=seed_sr,
                                  max_events=max_events_strict)
    println("done")
end
println()

# ------------------------------------------------------------------------------
# RESULTS: MIXED PRECISION
# ------------------------------------------------------------------------------

println("="^80)
println("RESULTS: End-Time Protein A Statistics (MIXED)")
println("="^80)
println()

println("Precision Format    |    Mean    |  Variance  |     n")
println("-"^60)
for (label, _, _) in mixed_configs
    c = results[label].end_pA
    @printf("%-19s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
end
println()

# ------------------------------------------------------------------------------
# RESULTS: STRICT PRECISION
# ------------------------------------------------------------------------------

println("="^80)
println("RESULTS: End-Time Protein A Statistics (STRICT)")
println("="^80)
println()

println("Precision Format    |    Mean    |  Variance  |     n")
println("-"^60)
for (label, _) in strict_configs
    c = results[label].end_pA
    @printf("%-19s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
end
println()

# ------------------------------------------------------------------------------
# RESULTS: WASSERSTEIN DISTANCES
# ------------------------------------------------------------------------------

println("="^80)
println("RESULTS: Wasserstein Distance vs FP64 (Protein A)")
println("="^80)
println()

ref = results["FP64 baseline"].end_pA

println("MIXED Precision:")
println("Precision Format    |  Wasserstein Distance")
println("-"^50)
for (label, _, _) in mixed_configs
    label == "FP64 baseline" && continue
    w = wasserstein_distance(ref, results[label].end_pA)
    @printf("%-19s |  %18.6f\n", label * " vs FP64", w)
end
println()

println("STRICT Precision:")
println("Precision Format    |  Wasserstein Distance")
println("-"^50)
for (label, _) in strict_configs
    w = wasserstein_distance(ref, results[label].end_pA)
    @printf("%-19s |  %18.6f\n", label * " vs FP64", w)
end
println()

# ------------------------------------------------------------------------------
# RESULTS: OSCILLATION FIDELITY
# ------------------------------------------------------------------------------

println("="^80)
println("RESULTS: Oscillation Fidelity (Protein A)")
println("="^80)
println()

println("Precision Format      |  Mean Peaks  |  Mean Amplitude")
println("-"^60)
for (label, _, _) in mixed_configs
    r = results[label]
    @printf("%-21s | %12.2f | %15.2f\n", label, mean(r.peak_counts), mean(r.peak_amplitudes))
end
println("-- STRICT --")
for (label, _) in strict_configs
    r = results[label]
    @printf("%-21s | %12.2f | %15.2f\n", label, mean(r.peak_counts), mean(r.peak_amplitudes))
end
println()

# ------------------------------------------------------------------------------
# GENERATE PLOTS
# ------------------------------------------------------------------------------

println("="^80)
println("Generating Plots...")
println("="^80)
println()

if skip_plots
    println("  Plot generation skipped (--skip-plots)")
else
    # Plot 1: Single trajectory overlay (FP64 vs BF16+SR)
    println("  [1/2] Representative trajectory overlay...")

    dual_fp64 = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual_fp64)
    traj_fp64 = ssa_repressilator!(model, t_end; Tprop=Float64, Tacc=Float64, rng=dual_fp64.ssa, mode=:mixed)

    dual_bfsr = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual_bfsr)
    traj_bfsr = ssa_repressilator!(model, t_end; Tprop=BFloat16sr, Tacc=Float32, rng=dual_bfsr.ssa, mode=:mixed)

    p1 = plot(xlabel="Time", ylabel="Protein A count",
              title="Repressilator: Protein A trajectory (FP64 vs BF16+SR)", legend=:topright)
    plot!(p1, traj_fp64.times, traj_fp64.counts_pA; label="FP64", alpha=0.7, lw=1)
    plot!(p1, traj_bfsr.times, traj_bfsr.counts_pA; label="BF16 + SR", alpha=0.7, lw=1, linestyle=:dash)

    savefig(p1, "figures/repressilator_trajectory.png")
    println("      Saved: figures/repressilator_trajectory.png")

    # Plot 2: End-time protein A distributions
    println("  [2/2] End-time protein A distributions...")

    data_all = [results[l].end_pA for (l, _, _) in mixed_configs]
    labels = [l for (l, _, _) in mixed_configs]

    lo_all = minimum(minimum.(data_all))
    hi_all = maximum(maximum.(data_all))
    edges_all = collect((lo_all - 0.5):1:(hi_all + 0.5))
    centers_all = (edges_all[1:end-1] .+ edges_all[2:end]) ./ 2

    p2 = plot(xlabel="Protein A count", ylabel="Probability",
              title="Repressilator: end-time protein A distribution (Mixed precision)", legend=:topright)

    for (data, lbl) in zip(data_all, labels)
        h = fit(Histogram, data, edges_all)
        probs = h.weights ./ sum(h.weights)
        bar!(p2, centers_all, probs; alpha=0.5, label=lbl)
    end

    savefig(p2, "figures/repressilator_protein_dist.png")
    println("      Saved: figures/repressilator_protein_dist.png")
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

# Oscillation stats as extra metrics
osc_extra = Dict{String, Any}()
for (label, _) in Iterators.flatten((mixed_configs, strict_configs))
    haskey(results, label) || continue
    r = results[label]
    osc_extra[label] = Dict{String, Any}(
        "mean_peaks"     => mean(r.peak_counts),
        "std_peaks"      => std(r.peak_counts),
        "mean_amplitude" => mean(r.peak_amplitudes),
        "std_amplitude"  => std(r.peak_amplitudes),
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
    count_field=:end_pA,
    extra_metrics=osc_extra,
    output_dir=output_dir,
    extra_metadata=meta_extra)

println()
println("="^80)
println("Analysis Complete!")
println("="^80)
println()
