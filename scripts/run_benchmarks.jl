#!/usr/bin/env julia

"""
LowPrecBio -- Benchmark Harness

Measures wall-clock time, memory allocation, events/second throughput, and
ensemble throughput for all 5 models across 6 mixed-precision configurations.

Results are saved to `results/benchmarks/` in JLD2 + CSV format.
"""

using LowPrecBio
using Printf

# ==============================================================================
# Precision configurations (mixed-mode only)
# ==============================================================================

# (label, prop_precision, acc_precision, mode)
const CONFIGS = [
    ("FP64 baseline",  :fp64,     :fp64, :mixed),
    ("FP32",           :fp32,     :fp32, :mixed),
    ("BF16 + SR",      :bf16_sr,  :fp32, :mixed),
    ("FP16 + SR",      :fp16_sr,  :fp32, :mixed),
    ("BF16 RTN",       :bf16_rtn, :fp32, :mixed),
    ("FP16 RTN",       :fp16_rtn, :fp32, :mixed),
]

# ==============================================================================
# Model registry
# ==============================================================================

# Each entry: (name, ssa_fn, model_instance, t_end, n_bench_replicas)
const MODELS = [
    (
        "birth_death",
        ssa_birth_death!,
        BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0),
        200.0,
        1000,
    ),
    (
        "schlogl",
        ssa_schlogl!,
        SchloglModel(k1=3e-7, k2=1e-4, k3=1.0, k4=3.5, A=2e5, B=2e5, initial_population=250),
        5.0,
        200,
    ),
    (
        "telegraph",
        ssa_telegraph!,
        TelegraphModel(k_on=0.01, k_off=0.1, alpha=5.0, beta=1.0, initial_population=0, initial_state=0),
        500.0,
        1000,
    ),
    (
        "dimer",
        ssa_dimer!,
        DimerModel(kf=1e-3, kr=0.1, initial_A=100, initial_D=0),
        400.0,
        1000,
    ),
    (
        "repressilator",
        ssa_repressilator!,
        RepressilatorModel(alpha0=1.0, alpha=216.0, n=2, delta_m=1.0, beta=5.0, delta_p=1.0,
                           initial_mA=0, initial_mB=0, initial_mC=0,
                           initial_pA=5, initial_pB=0, initial_pC=0),
        200.0,
        200,
    ),
]

# ==============================================================================
# Formatting helpers
# ==============================================================================

function print_model_header(name, model, t_end, n_bench)
    println()
    println("="^80)
    @printf("  Model: %-20s  t_end=%.1f  ensemble=%d\n", name, t_end, n_bench)
    println("="^80)
end

function print_results_table(bench_results)
    println()
    @printf("  %-19s │ %11s │ %10s │ %12s │ %12s\n",
            "Config", "Median (ms)", "Memory (KB)", "Events/s", "Ens. tput")
    println("  ", "─"^19, "─┼─", "─"^11, "─┼─", "─"^10, "─┼─", "─"^12, "─┼─", "─"^12)
    for label in sort(collect(keys(bench_results)))
        br = bench_results[label]
        @printf("  %-19s │ %11.3f │ %10.1f │ %12.0f │ %12.1f\n",
                br.label,
                br.median_time_ns / 1e6,
                br.memory_bytes / 1024.0,
                br.events_per_second,
                br.ensemble_throughput)
    end
    println()
end

# ==============================================================================
# Main
# ==============================================================================

println("="^80)
println("LowPrecBio -- Benchmark Harness")
println("="^80)
println()
println("Models:   $(length(MODELS))")
println("Configs:  $(length(CONFIGS))")
println()

all_results = Dict{String, Dict{String, BenchmarkResult}}()

for (name, ssa_fn, model, t_end, n_bench) in MODELS
    print_model_header(name, model, t_end, n_bench)

    println()
    for (i, (label, _, _, _)) in enumerate(CONFIGS)
        @printf("  [%d/%d] %-19s ...\n", i, length(CONFIGS), label)
    end
    println()
    println("  Running benchmarks (this may take a while)...")
    println()

    bench = benchmark_model(ssa_fn, model, t_end, CONFIGS;
                            n_bench_replicas=n_bench,
                            samples=10, seconds=30)

    all_results[name] = bench
    print_results_table(bench)

    # Save per-model results
    save_benchmark_results(name, bench)
end

# Save cross-model summary
println()
println("="^80)
println("Saving cross-model summary...")
println("="^80)
save_benchmark_summary(all_results)

println()
println("="^80)
println("Benchmarking Complete!")
println("="^80)
println()
