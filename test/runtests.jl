using Test

@testset "LowPrecBio.jl" begin
    @testset "Precision utilities" begin
        include("test_precision.jl")
    end

    @testset "Model implementations" begin
        include("test_models.jl")
    end

    @testset "Statistical utilities" begin
        include("test_statistics.jl")
    end

    @testset "I/O and serialization" begin
        include("test_io.jl")
    end

    @testset "Benchmarking utilities" begin
        include("test_benchmarks.jl")
    end
end
