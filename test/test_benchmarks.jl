using LowPrecBio
using Test

@testset "BenchmarkResult construction" begin
    br = BenchmarkResult(
        label="test", precision=:fp64, accumulator=:fp64, mode=:mixed,
        median_time_ns=1e6, mean_time_ns=1.1e6, min_time_ns=0.9e6,
        memory_bytes=1024, allocs=10,
        n_events_typical=100, events_per_second=1e5,
        ensemble_size=10, ensemble_total_s=1.0,
        ensemble_per_replica_s=0.1, ensemble_throughput=10.0
    )
    @test br.label == "test"
    @test br.precision == :fp64
    @test br.median_time_ns == 1e6
    @test br.ensemble_throughput == 10.0
end

@testset "benchmark_single" begin
    model = BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0)
    result = benchmark_single(ssa_birth_death!, model, 1.0;
                              prec=:fp64, acc=:fp64, samples=3, seconds=5)
    @test result.median_time_ns > 0
    @test result.mean_time_ns > 0
    @test result.min_time_ns > 0
    @test result.memory_bytes >= 0
    @test result.allocs >= 0
    @test result.n_events_typical > 0
    @test result.events_per_second > 0
end

@testset "benchmark_ensemble" begin
    model = BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0)
    result = benchmark_ensemble(ssa_birth_death!, model, 1.0, 5;
                                prec=:fp64, acc=:fp64)
    @test result.ensemble_size == 5
    @test result.ensemble_total_s > 0
    @test result.ensemble_per_replica_s > 0
    @test result.ensemble_throughput > 0
end

@testset "benchmark_model" begin
    model = BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0)
    configs = [
        ("FP64", :fp64, :fp64, :mixed),
        ("FP32", :fp32, :fp32, :mixed),
    ]
    results = benchmark_model(ssa_birth_death!, model, 1.0, configs;
                              n_bench_replicas=5, samples=3, seconds=5)
    @test length(results) == 2
    @test haskey(results, "FP64")
    @test haskey(results, "FP32")
    @test results["FP64"] isa BenchmarkResult
    @test results["FP32"].events_per_second > 0
end

@testset "save_benchmark_results" begin
    mktempdir() do dir
        br = BenchmarkResult(
            label="FP64", precision=:fp64, accumulator=:fp64, mode=:mixed,
            median_time_ns=1e6, mean_time_ns=1.1e6, min_time_ns=0.9e6,
            memory_bytes=1024, allocs=10,
            n_events_typical=100, events_per_second=1e5,
            ensemble_size=10, ensemble_total_s=1.0,
            ensemble_per_replica_s=0.1, ensemble_throughput=10.0
        )
        bench = Dict{String, BenchmarkResult}("FP64" => br)

        outdir = save_benchmark_results("test_model", bench; output_dir=dir)
        @test isdir(outdir)
        @test isfile(joinpath(outdir, "benchmarks.jld2"))
        @test isfile(joinpath(outdir, "benchmarks.csv"))

        # Verify CSV has header + 1 data row
        lines = readlines(joinpath(outdir, "benchmarks.csv"))
        @test length(lines) == 2
        @test startswith(lines[1], "label,")
    end
end
