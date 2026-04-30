using LowPrecBio
using JLD2
using Test

@testset "model_to_dict" begin
    @testset "BirthDeathModel" begin
        m = BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0)
        d = model_to_dict(m)
        @test d isa Dict{String, Any}
        @test d["birth_rate"] == 10.0
        @test d["death_rate"] == 0.5
        @test d["initial_population"] == 0
        @test length(d) == 3
    end

    @testset "SchloglModel" begin
        m = SchloglModel(k1=3e-7, k2=1e-4, k3=1.0, k4=3.5, A=2e5, B=2e5, initial_population=250)
        d = model_to_dict(m)
        @test d["k1"] == 3e-7
        @test d["initial_population"] == 250
        @test length(d) == 7
    end

    @testset "TelegraphModel" begin
        m = TelegraphModel()
        d = model_to_dict(m)
        @test d["k_on"] == 0.01
        @test d["k_off"] == 0.1
        @test length(d) == 6
    end

    @testset "DimerModel" begin
        m = DimerModel(kf=1e-3, kr=0.1, initial_A=100, initial_D=0)
        d = model_to_dict(m)
        @test d["kf"] == 1e-3
        @test d["initial_A"] == 100
        @test length(d) == 4
    end

    @testset "RepressilatorModel" begin
        m = RepressilatorModel()
        d = model_to_dict(m)
        @test d["alpha0"] == 1.0
        @test d["n"] == 2
        @test length(d) == 12
    end
end

@testset "compute_validation_metrics" begin
    # Create synthetic results with known properties
    ref_data = collect(1:100)
    test_data = collect(2:101)  # shifted by 1
    same_data = collect(1:100)

    results = Dict{String, NamedTuple}(
        "reference" => (; counts=ref_data),
        "shifted"   => (; counts=test_data),
        "same"      => (; counts=same_data),
    )

    metrics = compute_validation_metrics(results, "reference"; count_field=:counts)

    @test haskey(metrics, "reference")
    @test haskey(metrics, "shifted")
    @test haskey(metrics, "same")

    # Self-comparison should have zero Wasserstein distance
    @test metrics["reference"]["wasserstein"] == 0.0

    # Same data should also have zero distance
    @test metrics["same"]["wasserstein"] == 0.0

    # Shifted data should have nonzero distance
    @test metrics["shifted"]["wasserstein"] > 0.0

    # Check expected keys exist
    for key in ["wasserstein", "ks_statistic", "ks_pvalue", "mean", "mean_lower",
                "mean_upper", "variance", "var_lower", "var_upper", "n_negative", "n_total"]
        @test haskey(metrics["shifted"], key)
    end

    @test metrics["shifted"]["n_negative"] == 0
    @test metrics["shifted"]["n_total"] == 100
end

@testset "compute_validation_metrics multi-field" begin
    results = Dict{String, NamedTuple}(
        "ref"  => (; A=collect(1:50), D=collect(10:59)),
        "test" => (; A=collect(2:51), D=collect(10:59)),
    )

    metrics = compute_validation_metrics(results, "ref"; count_field=[:A, :D])

    @test haskey(metrics["test"], "A_wasserstein")
    @test haskey(metrics["test"], "D_wasserstein")
    @test metrics["test"]["A_wasserstein"] > 0.0
    @test metrics["test"]["D_wasserstein"] == 0.0
end

@testset "save/load ensemble_data round-trip" begin
    mktempdir() do dir
        filepath = joinpath(dir, "test_ensemble.jld2")

        original = Dict{String, NamedTuple}(
            "FP64" => (; counts=collect(1:10)),
            "FP16" => (; counts=collect(2:11)),
        )

        LowPrecBio.save_ensemble_data(filepath, original)
        loaded = load_ensemble_data(filepath)

        @test loaded isa Dict
        @test haskey(loaded, "FP64")
        @test haskey(loaded, "FP16")
        @test loaded["FP64"]["counts"] == collect(1:10)
        @test loaded["FP16"]["counts"] == collect(2:11)
    end
end

@testset "LowPrecBio.save_summary_csv" begin
    mktempdir() do dir
        filepath = joinpath(dir, "test_summary.csv")

        metrics = Dict{String, Dict{String, Any}}(
            "FP64" => Dict{String, Any}("label" => "FP64", "wasserstein" => 0.0, "mean" => 5.5, "n_total" => 100),
            "FP16" => Dict{String, Any}("label" => "FP16", "wasserstein" => 1.2, "mean" => 6.0, "n_total" => 100),
        )

        LowPrecBio.save_summary_csv(filepath, metrics; labels_order=["FP64", "FP16"])

        lines = readlines(filepath)
        @test length(lines) == 3  # header + 2 data rows
        @test startswith(lines[1], "label,")
        @test startswith(lines[2], "FP64,")
        @test startswith(lines[3], "FP16,")
    end
end

@testset "save_results creates directory structure" begin
    mktempdir() do dir
        results = Dict{String, NamedTuple}(
            "FP64 baseline" => (; counts=collect(1:20)),
            "FP16 + SR"     => (; counts=collect(2:21)),
        )
        model = BirthDeathModel(birth_rate=10.0, death_rate=0.5, initial_population=0)
        configs = [
            ("FP64 baseline", :fp64, :fp64, :mixed),
            ("FP16 + SR", :fp16_sr, :fp32, :mixed),
        ]

        outdir = save_results("birth_death", results;
            model=model,
            t_end=200.0,
            n_replicas=20,
            seed_ssa=42,
            seed_sr=4242,
            configs=configs,
            count_field=:counts,
            output_dir=dir)

        @test isdir(outdir)
        @test isfile(joinpath(outdir, "ensemble_data.jld2"))
        @test isfile(joinpath(outdir, "metadata.jld2"))
        @test isfile(joinpath(outdir, "summary_stats.csv"))

        # Verify metadata round-trip
        meta = JLD2.load(joinpath(outdir, "metadata.jld2"), "metadata")
        @test meta["model_name"] == "birth_death"
        @test meta["n_replicas"] == 20
        @test meta["seed_ssa"] == 42

        # Verify ensemble round-trip
        ens = load_ensemble_data(joinpath(outdir, "ensemble_data.jld2"))
        @test ens["FP64 baseline"]["counts"] == collect(1:20)
    end
end

@testset "save_results with extra_metrics" begin
    mktempdir() do dir
        results = Dict{String, NamedTuple}(
            "FP64 baseline" => (; counts=collect(1:20)),
        )
        model = BirthDeathModel()
        configs = [("FP64 baseline", :fp64, :fp64, :mixed)]

        extra = Dict{String, Any}("key1" => 42, "key2" => [1.0, 2.0])

        save_results("test_extra", results;
            model=model,
            t_end=100.0,
            n_replicas=20,
            seed_ssa=1,
            seed_sr=2,
            configs=configs,
            count_field=:counts,
            extra_metrics=extra,
            output_dir=dir)

        @test isfile(joinpath(dir, "test_extra", "extra_metrics.jld2"))
        loaded_extra = JLD2.load(joinpath(dir, "test_extra", "extra_metrics.jld2"), "extra")
        @test loaded_extra["key1"] == 42
        @test loaded_extra["key2"] == [1.0, 2.0]
    end
end
