#!/usr/bin/env julia

"""
LowPrecBio -- Schlogl Model with Mixed and Strict Low Precision (with SR)

This script runs precision-sweep experiments for the **Schlogl model**, a bistable
autocatalytic system. Tests both mixed precision and strict precision modes.

Uses the LowPrecBio library for models, SSA, and analysis utilities.
"""

using LowPrecBio
using Random, Statistics, StatsBase
using StochasticRounding
using Printf
include("cli_args.jl")

# ==============================================================================
# Script-specific helpers
# ==============================================================================

"""
    sr_variability_test(T; N=50_000)

SR sanity check: convert same vector twice, measure fraction of differing elements.
SR types should show > 0 variability; RTN types should show ~0.
"""
function sr_variability_test(T; N=50_000)
    v = rand(Float64, N) .* 2 .- 1
    a = T.(v)
    b = T.(v)
    return mean(Float64.(a) .!= Float64.(b))
end

# ==============================================================================
# Ensemble runner
# ==============================================================================

"""
    run_ensemble(model, t_end, n_replicas, prec, acc; mode=:mixed)

Run `n_replicas` independent SSA trajectories and collect end-time molecule counts.
The `mode` kwarg selects mixed vs strict precision in the library SSA.
"""
function run_ensemble(model::SchloglModel, t_end, n_replicas, prec::Symbol, acc::Symbol;
                      mode::Symbol=:mixed, seed_ssa=777, seed_sr=2025,
                      max_events::Int=typemax(Int))
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)
    dual  = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual)

    counts = Vector{Int}(undef, n_replicas)

    @inbounds for i in 1:n_replicas
        result = ssa_schlogl!(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=dual.ssa,
                              mode=mode, max_events=max_events)
        counts[i] = result.counts[end]
        # Periodically free trajectory memory (Schlogl has high event rates → large trajectories)
        i % 200 == 0 && GC.gc(true)
    end

    return (; counts)
end

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_cli_args(ARGS)

    seed_ssa = arg_int(opts, "seed-ssa", 777)
    seed_sr = arg_int(opts, "seed-sr", 2025)
    t_end = arg_float(opts, "t-end", 10.0)
    n_replicas = arg_int(opts, "n-replicas", 10_000)
    # Rate constants corrected for propensity formulation without combinatorial factors:
    # Code uses a1 = k1*A*n*(n-1) and a2 = k2*n*(n-1)*(n-2), i.e. no 1/2! or 1/3! divisors.
    # Literature (Cao & Petzold 2005) quotes c1=3e-7, c2=1e-4 WITH those divisors,
    # so k1 = c1/2 and k2 = c2/6 to match the same effective propensities.
    k1 = arg_float(opts, "k1", 1.5e-7)
    k2 = arg_float(opts, "k2", 1e-4/6)
    k3 = arg_float(opts, "k3", 1e-3)
    k4 = arg_float(opts, "k4", 3.5)
    A = arg_float(opts, "A", 1e5)
    B = arg_float(opts, "B", 2e5)
    initial_population = arg_int(opts, "initial-population", 250)
    output_dir = arg_string(opts, "output-dir", "results")
    tag = arg_string(opts, "tag", "")
    skip_strict = arg_bool(opts, "skip-strict", false)
    strict_rtn = arg_bool(opts, "strict-rtn", true)
    mixed_set = lowercase(arg_string(opts, "mixed-set", "safe"))
    model_name = model_name_with_tag("schlogl", tag)

    println("="^80)
    println("Schlogl Model -- Mixed vs Strict Low Precision (with SR)")
    println("="^80)

    # Classic bistable Schlogl parameters (Cao & Petzold 2005, corrected for propensity formulation)
    model = SchloglModel(
        k1=k1, k2=k2, k3=k3, k4=k4,
        A=A, B=B,
        initial_population=initial_population
    )
    println("Simulation Configuration:")
    println("-"^80)
    println("  Simulation time:  $t_end")
    println("  Ensemble size:    $n_replicas trajectories")
    println("  Seeds:            seed_ssa=$seed_ssa, seed_sr=$seed_sr")
    println("  Output dir:       $output_dir")
    println("  Model name:       $model_name")
    println()

    # Mixed precision configurations: (label, prop_precision, acc_precision)
    mixed_configs = if mixed_set == "safe"
        [
            ("FP64 baseline",  :fp64,     :fp64),
            ("FP32",           :fp32,     :fp32),
            ("BF16 + SR",      :bf16_sr,  :fp32),
            ("BF16 RTN",       :bf16_rtn, :fp32),
        ]
    elseif mixed_set == "minimal"
        [
            ("FP64 baseline",  :fp64,     :fp64),
            ("BF16 + SR",      :bf16_sr,  :fp32),
        ]
    elseif mixed_set == "all"
        [
            ("FP64 baseline",  :fp64,     :fp64),
            ("FP32",           :fp32,     :fp32),
            ("BF16 + SR",      :bf16_sr,  :fp32),
            ("FP16 + SR",      :fp16_sr,  :fp32),
            ("BF16 RTN",       :bf16_rtn, :fp32),
            ("FP16 RTN",       :fp16_rtn, :fp32),
        ]
    else
        error("Invalid --mixed-set value '$mixed_set' (expected: all, safe, minimal)")
    end

    # Strict precision configurations: (label, prop_precision)
    # Note: STRICT BF16 RTN causes population runaway due to RTN rounding bias on cubic
    # propensities (413M events at t=1.0, population reaches 768). Excluded by default.
    strict_configs = Tuple{String, Symbol}[
        ("STRICT BF16 + SR",  :bf16_sr),
        ("STRICT FP16 + SR",  :fp16_sr),
    ]
    if strict_rtn
        push!(strict_configs, ("STRICT BF16 RTN", :bf16_rtn))
        push!(strict_configs, ("STRICT FP16 RTN", :fp16_rtn))
    end

    # Run mixed precision experiments
    println("Running MIXED-precision ensembles...")
    results = Dict{String, NamedTuple}()
    for (label, prec, acc) in mixed_configs
        print("  $label... ")
        results[label] = run_ensemble(model, t_end, n_replicas, prec, acc;
                                      mode=:mixed, seed_ssa=seed_ssa, seed_sr=seed_sr)
        println("done")
    end

    # Run strict precision experiments
    if skip_strict
        println("Skipping STRICT low-precision ensembles (--skip-strict)")
    else
        println("Running STRICT low-precision ensembles...")
        # max_events guard prevents runaway memory from FP16 overflow or BF16 RTN divergence
        strict_max_events = 1_000_000
        for (label, prec) in strict_configs
            print("  $label... ")
            results[label] = run_ensemble(model, t_end, n_replicas, prec, :fp32;
                                          mode=:strict, seed_ssa=seed_ssa, seed_sr=seed_sr,
                                          max_events=strict_max_events)
            println("done")
        end
    end

    # Summaries
    println()
    println("="^80)
    println("RESULTS: End-Time Molecule Count Statistics (MIXED)")
    println("="^80)
    @printf("%-18s | %10s | %10s | %6s\n", "Precision", "Mean", "Variance", "n")
    println("-"^60)
    for (label, _, _) in mixed_configs
        c = results[label].counts
        @printf("%-18s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
    end

    if !skip_strict
        println()
        println("="^80)
        println("RESULTS: End-Time Molecule Count Statistics (STRICT)")
        println("="^80)
        @printf("%-18s | %10s | %10s | %6s\n", "Precision", "Mean", "Variance", "n")
        println("-"^60)
        for (label, _) in strict_configs
            c = results[label].counts
            @printf("%-18s | %10.4f | %10.4f | %6d\n", label, mean(c), var(c), length(c))
        end
    end

    # Wasserstein distances
    ref = results["FP64 baseline"].counts

    println()
    println("="^80)
    println("RESULTS: Wasserstein Distance vs FP64")
    println("="^80)
    for (label, _, _) in mixed_configs
        label == "FP64 baseline" && continue
        w = wasserstein_distance(ref, results[label].counts)
        @printf("%-22s | %12.6f\n", label, w)
    end
    if !skip_strict
        println("-- STRICT --")
        for (label, _) in strict_configs
            w = wasserstein_distance(ref, results[label].counts)
            @printf("%-22s | %12.6f\n", label, w)
        end
    end

    # SR variability notes
    sr_bfsr = sr_variability_test(BFloat16sr)
    sr_fpsr = sr_variability_test(Float16sr)
    @printf("\nSR variability (fraction of elements changed on reconversion):\n")
    @printf("  BF16+SR: %.4f\n", sr_bfsr)
    @printf("  FP16+SR: %.4f\n", sr_fpsr)

    # Save results
    println()
    println("="^80)
    println("Saving Results...")
    println("="^80)
    println()

    all_configs = [(l, p, a, :mixed) for (l, p, a) in mixed_configs]
    if !skip_strict
        append!(all_configs, [(l, p, :fp32, :strict) for (l, p) in strict_configs])
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
        output_dir=output_dir,
        extra_metadata=meta_extra)

    println("\nDone.")
end
