#!/usr/bin/env julia

"""
LowPrecBio -- Run all strict-mode experiments for all models.

This script runs strict-precision experiments that were previously missing:
- Birth-Death: 4 strict modes (all new)
- Dimer: 4 strict modes (all new)
- Schlogl: FP16+SR strict (new), FP16 RTN strict (new)
- Repressilator: BF16 RTN strict, FP16 RTN strict (new, with max_events guard)

Does NOT import Plots (avoids REPL/PTY sandbox issues).
Results are saved to the output directory for each model.
"""

using LowPrecBio
using Random, Statistics, StatsBase
using StochasticRounding
using Printf

include("cli_args.jl")

# ==============================================================================
# HELPERS
# ==============================================================================

function print_header(title)
    println("="^80)
    println(title)
    println("="^80)
    println()
end

function print_stats(label, counts; ref_counts=nothing)
    m = mean(counts)
    v = var(counts)
    @printf("  %-22s | mean=%10.4f | var=%10.4f | n=%d", label, m, v, length(counts))
    if ref_counts !== nothing
        w = wasserstein_distance(ref_counts, counts)
        @printf(" | W1=%8.4f", w)
    end
    println()
end

# ==============================================================================
# 1. BIRTH-DEATH (strict modes)
# ==============================================================================

print_header("1. BIRTH-DEATH: Strict-Mode Experiments")

bd_model = BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0)
bd_t_end = 200.0
bd_n_replicas = 50_000
bd_seed_ssa = 42
bd_seed_sr = 4242

# Run FP64 baseline for comparison
println("  Running FP64 baseline...")
let
    dual = DualRNG(bd_seed_ssa, bd_seed_sr)
    activate_sr_rng!(dual)
    global bd_ref_counts = Vector{Int}(undef, bd_n_replicas)
    @inbounds for i in 1:bd_n_replicas
        result = ssa_birth_death!(bd_model, bd_t_end; Tprop=Float64, Tacc=Float64, rng=dual.ssa, mode=:mixed)
        bd_ref_counts[i] = result.counts[end]
    end
end

bd_strict_configs = [
    ("STRICT BF16 + SR",  :bf16_sr),
    ("STRICT FP16 + SR",  :fp16_sr),
    ("STRICT BF16 RTN",   :bf16_rtn),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

bd_strict_results = Dict{String, Vector{Int}}()
for (label, prec) in bd_strict_configs
    print("  $label... ")
    Tprop = precision_type(prec)
    dual = DualRNG(bd_seed_ssa, bd_seed_sr)
    activate_sr_rng!(dual)
    counts = Vector{Int}(undef, bd_n_replicas)
    @inbounds for i in 1:bd_n_replicas
        result = ssa_birth_death!(bd_model, bd_t_end; Tprop=Tprop, Tacc=Float32, rng=dual.ssa, mode=:strict)
        counts[i] = result.counts[end]
    end
    bd_strict_results[label] = counts
    println("done")
end

println("\nResults:")
for (label, _) in bd_strict_configs
    print_stats(label, bd_strict_results[label]; ref_counts=bd_ref_counts)
end
println()

# ==============================================================================
# 2. DIMER (strict modes)
# ==============================================================================

print_header("2. DIMER: Strict-Mode Experiments")

dimer_model = DimerModel(kf=1e-3, kr=0.1, initial_A=100, initial_D=0)
dimer_t_end = 400.0
dimer_n_replicas = 50_000
dimer_seed_ssa = 101
dimer_seed_sr = 2026

# Run FP64 baseline
println("  Running FP64 baseline...")
let
    dual = DualRNG(dimer_seed_ssa, dimer_seed_sr)
    activate_sr_rng!(dual)
    global dimer_ref_A = Vector{Int}(undef, dimer_n_replicas)
    global dimer_ref_D = Vector{Int}(undef, dimer_n_replicas)
    @inbounds for i in 1:dimer_n_replicas
        result = ssa_dimer!(dimer_model, dimer_t_end; Tprop=Float64, Tacc=Float64, rng=dual.ssa, mode=:mixed)
        dimer_ref_A[i] = result.counts_A[end]
        dimer_ref_D[i] = result.counts_D[end]
    end
end

dimer_strict_configs = [
    ("STRICT BF16 + SR",  :bf16_sr),
    ("STRICT FP16 + SR",  :fp16_sr),
    ("STRICT BF16 RTN",   :bf16_rtn),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

dimer_strict_A = Dict{String, Vector{Int}}()
dimer_strict_D = Dict{String, Vector{Int}}()
dimer_strict_dev = Dict{String, Vector{Int}}()
for (label, prec) in dimer_strict_configs
    print("  $label... ")
    Tprop = precision_type(prec)
    dual = DualRNG(dimer_seed_ssa, dimer_seed_sr)
    activate_sr_rng!(dual)
    As = Vector{Int}(undef, dimer_n_replicas)
    Ds = Vector{Int}(undef, dimer_n_replicas)
    devs = Vector{Int}(undef, dimer_n_replicas)
    @inbounds for i in 1:dimer_n_replicas
        result = ssa_dimer!(dimer_model, dimer_t_end; Tprop=Tprop, Tacc=Float32, rng=dual.ssa, mode=:strict)
        As[i] = result.counts_A[end]
        Ds[i] = result.counts_D[end]
        devs[i] = result.max_deviation
    end
    dimer_strict_A[label] = As
    dimer_strict_D[label] = Ds
    dimer_strict_dev[label] = devs
    println("done")
end

println("\nResults (Monomer A):")
for (label, _) in dimer_strict_configs
    w_A = wasserstein_distance(dimer_ref_A, dimer_strict_A[label])
    w_D = wasserstein_distance(dimer_ref_D, dimer_strict_D[label])
    frac_exact = count(==(0), dimer_strict_dev[label]) / length(dimer_strict_dev[label])
    @printf("  %-22s | mean(A)=%7.3f | var(A)=%7.3f | W1(A)=%7.4f | W1(D)=%7.4f | conserv=%.4f\n",
            label, mean(dimer_strict_A[label]), var(dimer_strict_A[label]), w_A, w_D, frac_exact)
end
println()

# ==============================================================================
# 3. SCHLOGL (new strict modes: FP16+SR, FP16 RTN)
# ==============================================================================

print_header("3. SCHLOGL: Additional Strict-Mode Experiments")

schlogl_model = SchloglModel(
    k1=1.5e-7, k2=1e-4/6, k3=1e-3, k4=3.5,
    A=1e5, B=2e5,
    initial_population=250
)
schlogl_t_end = 10.0
schlogl_n_replicas = 10_000
schlogl_seed_ssa = 777
schlogl_seed_sr = 2025

# Run FP64 baseline
println("  Running FP64 baseline...")
let
    dual = DualRNG(schlogl_seed_ssa, schlogl_seed_sr)
    activate_sr_rng!(dual)
    global schlogl_ref = Vector{Int}(undef, schlogl_n_replicas)
    @inbounds for i in 1:schlogl_n_replicas
        result = ssa_schlogl!(schlogl_model, schlogl_t_end; Tprop=Float64, Tacc=Float64, rng=dual.ssa, mode=:mixed)
        schlogl_ref[i] = result.counts[end]
        i % 200 == 0 && GC.gc(true)
    end
end

# FP16 strict modes will overflow (cubic propensities reach ~10^7 > FP16 max 65504).
# The isfinite(a0) check in the SSA will break the loop early on overflow.
# max_events guard prevents runaway memory usage if overflow isn't caught.
schlogl_new_strict = [
    ("STRICT FP16 + SR",  :fp16_sr),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

schlogl_max_events = 1_000_000
schlogl_strict_results = Dict{String, Any}()
for (label, prec) in schlogl_new_strict
    print("  $label... ")
    Tprop = precision_type(prec)
    dual = DualRNG(schlogl_seed_ssa, schlogl_seed_sr)
    activate_sr_rng!(dual)
    counts = Vector{Int}(undef, schlogl_n_replicas)
    failed = false
    try
        @inbounds for i in 1:schlogl_n_replicas
            result = ssa_schlogl!(schlogl_model, schlogl_t_end; Tprop=Tprop, Tacc=Float32,
                                  rng=dual.ssa, mode=:strict, max_events=schlogl_max_events)
            counts[i] = result.counts[end]
            i % 200 == 0 && GC.gc(true)
        end
    catch e
        println("FAILED: $e")
        failed = true
    end
    if !failed
        schlogl_strict_results[label] = counts
        println("done")
    end
end

println("\nResults:")
for (label, _) in schlogl_new_strict
    if haskey(schlogl_strict_results, label)
        print_stats(label, schlogl_strict_results[label]; ref_counts=schlogl_ref)
    else
        println("  $label: FAILED (overflow or divergence)")
    end
end
println()

# ==============================================================================
# 4. REPRESSILATOR (strict RTN modes with max_events guard)
# ==============================================================================

print_header("4. REPRESSILATOR: Strict RTN Experiments (with max_events guard)")

repr_model = RepressilatorModel(
    alpha0=1.0, alpha=216.0, n=2,
    delta_m=1.0, beta=5.0, delta_p=1.0,
    initial_mA=0, initial_mB=0, initial_mC=0,
    initial_pA=5, initial_pB=0, initial_pC=0,
)
repr_t_end = 200.0
repr_n_replicas = 1_000
repr_seed_ssa = 123
repr_seed_sr = 3030
repr_max_events = 500_000

# Run FP64 baseline
println("  Running FP64 baseline...")
let
    dual = DualRNG(repr_seed_ssa, repr_seed_sr)
    activate_sr_rng!(dual)
    global repr_ref = Vector{Int}(undef, repr_n_replicas)
    @inbounds for i in 1:repr_n_replicas
        result = ssa_repressilator!(repr_model, repr_t_end; Tprop=Float64, Tacc=Float64, rng=dual.ssa, mode=:mixed)
        repr_ref[i] = result.counts_pA[end]
    end
end

repr_new_strict = [
    ("STRICT BF16 RTN",   :bf16_rtn),
    ("STRICT FP16 RTN",   :fp16_rtn),
]

repr_strict_results = Dict{String, Vector{Int}}()
for (label, prec) in repr_new_strict
    print("  $label... ")
    Tprop = precision_type(prec)
    dual = DualRNG(repr_seed_ssa, repr_seed_sr)
    activate_sr_rng!(dual)
    end_pA = Vector{Int}(undef, repr_n_replicas)
    @inbounds for i in 1:repr_n_replicas
        result = ssa_repressilator!(repr_model, repr_t_end; Tprop=Tprop, Tacc=Float32,
                                    rng=dual.ssa, mode=:strict, max_events=repr_max_events)
        end_pA[i] = result.counts_pA[end]
    end
    repr_strict_results[label] = end_pA
    println("done")
end

println("\nResults:")
for (label, _) in repr_new_strict
    print_stats(label, repr_strict_results[label]; ref_counts=repr_ref)
end
println()

# ==============================================================================
# SUMMARY
# ==============================================================================

print_header("SUMMARY: All New Strict-Mode Results")

println("Birth-Death strict modes:")
for (label, _) in bd_strict_configs
    print_stats(label, bd_strict_results[label]; ref_counts=bd_ref_counts)
end
println()

println("Dimer strict modes:")
for (label, _) in dimer_strict_configs
    w_A = wasserstein_distance(dimer_ref_A, dimer_strict_A[label])
    @printf("  %-22s | W1(A)=%7.4f\n", label, w_A)
end
println()

println("Schlogl new strict modes:")
for (label, _) in schlogl_new_strict
    if haskey(schlogl_strict_results, label)
        print_stats(label, schlogl_strict_results[label]; ref_counts=schlogl_ref)
    else
        println("  $label: FAILED (overflow/divergence)")
    end
end
println()

println("Repressilator strict RTN modes:")
for (label, _) in repr_new_strict
    print_stats(label, repr_strict_results[label]; ref_counts=repr_ref)
end

println()
println("="^80)
println("All experiments complete!")
println("="^80)
