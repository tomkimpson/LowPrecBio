# Benchmarking utilities for LowPrecBio
# Measures wall-clock time, memory allocation, events/second throughput,
# and ensemble throughput across models and precision modes.

using BenchmarkTools

"""
    BenchmarkResult

Container for benchmark measurements of a single model × precision configuration.

Identity fields identify the configuration; single-trajectory fields come from
`@benchmark`; ensemble fields come from a timed ensemble loop.
"""
Base.@kwdef struct BenchmarkResult
    # Identity
    label::String
    precision::Symbol
    accumulator::Symbol
    mode::Symbol

    # Single-trajectory metrics (from BenchmarkTools)
    median_time_ns::Float64
    mean_time_ns::Float64
    min_time_ns::Float64
    memory_bytes::Int
    allocs::Int
    n_events_typical::Int
    events_per_second::Float64

    # Ensemble metrics
    ensemble_size::Int
    ensemble_total_s::Float64
    ensemble_per_replica_s::Float64
    ensemble_throughput::Float64   # replicas / second
end

"""
    benchmark_single(ssa_fn, model, t_end; prec, acc, mode=:mixed, samples=10, seconds=30)

Benchmark a single SSA trajectory using BenchmarkTools.

Runs one trajectory first to get a representative `n_events`, then uses
`@benchmark` with `evals=1` (each SSA call is non-trivial and mutates RNG state).

Returns a named tuple with timing and allocation metrics.
"""
function benchmark_single(ssa_fn, model, t_end;
                          prec::Symbol, acc::Symbol, mode::Symbol=:mixed,
                          samples::Int=10, seconds::Real=30)
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)

    # Run one trajectory to get representative n_events
    warmup_rng = Xoshiro(9999)
    warmup_result = ssa_fn(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=warmup_rng, mode=mode)
    n_events = warmup_result.n_events

    # Benchmark with fresh RNG in setup
    b = @benchmark $ssa_fn($model, $t_end; Tprop=$Tprop, Tacc=$Tacc, rng=rng, mode=$mode) setup=(rng = Xoshiro(rand(UInt64))) evals=1 samples=samples seconds=seconds

    med_ns  = median(b.times)
    mean_ns = mean(b.times)
    min_ns  = minimum(b.times)
    mem     = b.memory
    allc    = b.allocs

    # events/second from median time
    eps = n_events / (med_ns / 1e9)

    return (median_time_ns=med_ns, mean_time_ns=mean_ns, min_time_ns=min_ns,
            memory_bytes=mem, allocs=allc, n_events_typical=n_events,
            events_per_second=eps)
end

"""
    benchmark_ensemble(ssa_fn, model, t_end, n_replicas; prec, acc, mode=:mixed)

Time a full ensemble of `n_replicas` SSA trajectories.

Does one warmup trajectory, then times the full loop with `@elapsed`.
Returns a named tuple with total time, per-replica time, and throughput.
"""
function benchmark_ensemble(ssa_fn, model, t_end, n_replicas::Int;
                            prec::Symbol, acc::Symbol, mode::Symbol=:mixed)
    Tprop = precision_type(prec)
    Tacc  = accum_type(acc)

    # Warmup
    warmup_rng = Xoshiro(9999)
    ssa_fn(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=warmup_rng, mode=mode)

    # Timed ensemble
    rng = Xoshiro(42)
    elapsed = @elapsed begin
        for _ in 1:n_replicas
            ssa_fn(model, t_end; Tprop=Tprop, Tacc=Tacc, rng=rng, mode=mode)
        end
    end

    per_replica = elapsed / n_replicas
    throughput  = n_replicas / elapsed

    return (ensemble_size=n_replicas, ensemble_total_s=elapsed,
            ensemble_per_replica_s=per_replica, ensemble_throughput=throughput)
end

"""
    benchmark_model(ssa_fn, model, t_end, configs; n_bench_replicas=1000, samples=10, seconds=30)

Run single-trajectory and ensemble benchmarks for each configuration in `configs`.

`configs` is a vector of tuples `(label, prec, acc, mode)`.

Returns `Dict{String, BenchmarkResult}` keyed by label.
"""
function benchmark_model(ssa_fn, model, t_end, configs;
                         n_bench_replicas::Int=1000,
                         samples::Int=10, seconds::Real=30)
    results = Dict{String, BenchmarkResult}()

    for (label, prec, acc, mode) in configs
        single = benchmark_single(ssa_fn, model, t_end;
                                  prec=prec, acc=acc, mode=mode,
                                  samples=samples, seconds=seconds)

        ens = benchmark_ensemble(ssa_fn, model, t_end, n_bench_replicas;
                                 prec=prec, acc=acc, mode=mode)

        results[label] = BenchmarkResult(
            label              = label,
            precision          = prec,
            accumulator        = acc,
            mode               = mode,
            median_time_ns     = single.median_time_ns,
            mean_time_ns       = single.mean_time_ns,
            min_time_ns        = single.min_time_ns,
            memory_bytes       = single.memory_bytes,
            allocs             = single.allocs,
            n_events_typical   = single.n_events_typical,
            events_per_second  = single.events_per_second,
            ensemble_size      = ens.ensemble_size,
            ensemble_total_s   = ens.ensemble_total_s,
            ensemble_per_replica_s = ens.ensemble_per_replica_s,
            ensemble_throughput    = ens.ensemble_throughput,
        )
    end

    return results
end

"""
    save_benchmark_results(model_name, bench_results; output_dir="results/benchmarks")

Save per-model benchmark results to `output_dir/<model_name>/benchmarks.jld2`
and `output_dir/<model_name>/benchmarks.csv`.
"""
function save_benchmark_results(model_name::String, bench_results::Dict{String, BenchmarkResult};
                                output_dir::String="results/benchmarks")
    dir = joinpath(output_dir, model_name)
    mkpath(dir)

    # Save JLD2 (convert structs to dicts for portability)
    data = Dict{String, Any}()
    for (label, br) in bench_results
        d = Dict{String, Any}()
        for f in fieldnames(BenchmarkResult)
            d[string(f)] = getfield(br, f)
        end
        data[label] = d
    end
    JLD2.save(joinpath(dir, "benchmarks.jld2"), "benchmarks", data)

    # Save CSV
    _write_benchmark_csv(joinpath(dir, "benchmarks.csv"), bench_results)

    println("Benchmark results saved to $(dir)/")
    println("  - benchmarks.jld2  ($(length(bench_results)) configs)")
    println("  - benchmarks.csv")

    return dir
end

"""
    save_benchmark_summary(all_results; output_dir="results/benchmarks")

Write a cross-model summary CSV at `output_dir/summary.csv`.

Columns: model, label, median_time_ms, memory_KB, events_per_second, ensemble_throughput
"""
function save_benchmark_summary(all_results::Dict{String, Dict{String, BenchmarkResult}};
                                output_dir::String="results/benchmarks")
    mkpath(output_dir)
    filepath = joinpath(output_dir, "summary.csv")

    open(filepath, "w") do io
        println(io, "model,label,precision,accumulator,mode,median_time_ms,memory_KB,events_per_second,ensemble_throughput")
        for model_name in sort(collect(keys(all_results)))
            bench = all_results[model_name]
            for label in sort(collect(keys(bench)))
                br = bench[label]
                @printf(io, "%s,%s,%s,%s,%s,%.4f,%.2f,%.1f,%.2f\n",
                        model_name, br.label, br.precision, br.accumulator, br.mode,
                        br.median_time_ns / 1e6,
                        br.memory_bytes / 1024.0,
                        br.events_per_second,
                        br.ensemble_throughput)
            end
        end
    end

    println("Summary saved to $(filepath)")
    return filepath
end

# --- Internal helpers ---

function _write_benchmark_csv(filepath::String, bench_results::Dict{String, BenchmarkResult})
    open(filepath, "w") do io
        println(io, "label,precision,accumulator,mode,median_time_ns,mean_time_ns,min_time_ns,memory_bytes,allocs,n_events_typical,events_per_second,ensemble_size,ensemble_total_s,ensemble_per_replica_s,ensemble_throughput")
        for label in sort(collect(keys(bench_results)))
            br = bench_results[label]
            @printf(io, "%s,%s,%s,%s,%.1f,%.1f,%.1f,%d,%d,%d,%.1f,%d,%.6f,%.6f,%.2f\n",
                    br.label, br.precision, br.accumulator, br.mode,
                    br.median_time_ns, br.mean_time_ns, br.min_time_ns,
                    br.memory_bytes, br.allocs, br.n_events_typical,
                    br.events_per_second,
                    br.ensemble_size, br.ensemble_total_s,
                    br.ensemble_per_replica_s, br.ensemble_throughput)
        end
    end
end
