# Tests for precision type utilities

using Test
using LowPrecBio
using Random: Xoshiro
using StochasticRounding: Float16sr, BFloat16, BFloat16sr

@testset "precision_type mappings" begin
    @testset "FP64 baseline" begin
        @test precision_type(:fp64) === Float64
    end

    @testset "FP32 standard" begin
        @test precision_type(:fp32) === Float32
    end

    @testset "FP16 variants" begin
        @test precision_type(:fp16_rtn) === Float16
        @test precision_type(:fp16_sr) === Float16sr
    end

    @testset "BFloat16 variants" begin
        @test precision_type(:bf16_rtn) === BFloat16
        @test precision_type(:bf16_sr) === BFloat16sr
    end

    @testset "Invalid precision throws" begin
        @test_throws ArgumentError precision_type(:invalid)
        @test_throws ArgumentError precision_type(:fp128)
    end
end

@testset "accum_type mappings" begin
    @testset "Valid accumulator types" begin
        @test accum_type(:fp64) === Float64
        @test accum_type(:fp32) === Float32
        @test accum_type(:fp16) === Float16
    end

    @testset "Invalid accumulator throws" begin
        @test_throws ArgumentError accum_type(:invalid)
        @test_throws ArgumentError accum_type(:bf16)
    end
end

@testset "Precision type properties" begin
    @testset "Type hierarchy" begin
        # Verify SR types are subtypes of AbstractFloat
        @test Float16sr <: AbstractFloat
        @test BFloat16sr <: AbstractFloat
    end

    @testset "Precision ordering" begin
        # Verify mantissa bit ordering (higher precision = more bits)
        @test precision(Float64) > precision(Float32) > precision(Float16)
    end
end

@testset "DualRNG" begin
    @testset "Default construction" begin
        d = DualRNG()
        @test d.ssa isa Xoshiro
        @test d.sr isa Xoshiro
    end

    @testset "Seeded construction" begin
        d = DualRNG(123, 456)
        @test d.ssa isa Xoshiro
        @test d.sr isa Xoshiro
    end

    @testset "Streams are independent" begin
        d = DualRNG(1, 1)
        # Even with same seed, drawing from one shouldn't affect the other
        v1 = rand(d.ssa)
        v2 = rand(d.sr)
        # Draw again from ssa — should differ from sr's first draw
        v3 = rand(d.ssa)
        @test v1 != v3  # ssa advanced
        # Reset and verify independence
        d2 = DualRNG(1, 1)
        _ = rand(d2.ssa)  # advance ssa only
        @test rand(d2.sr) == v2  # sr unchanged
    end

    @testset "activate_sr_rng! does not error" begin
        d = DualRNG()
        @test activate_sr_rng!(d) === nothing || true  # should not throw
    end
end

@testset "KahanAccumulator" begin
    @testset "Basic accumulation" begin
        acc = KahanAccumulator{Float64}()
        @test acc.s == 0.0
        @test acc.c == 0.0
        kahan_add!(acc, 1.0)
        @test acc.s == 1.0
        kahan_add!(acc, 2.0)
        @test acc.s == 3.0
    end

    @testset "Float32 compensation vs naive sum" begin
        # Sum many small values where naive Float32 loses precision
        acc = KahanAccumulator{Float32}()
        naive = Float32(0)
        n = 10_000
        val = Float32(0.0001)
        for _ in 1:n
            kahan_add!(acc, val)
            naive += val
        end
        exact = Float64(val) * n
        kahan_err = abs(Float64(acc.s) - exact)
        naive_err = abs(Float64(naive) - exact)
        @test kahan_err <= naive_err
    end

    @testset "Float16 accumulation" begin
        acc = KahanAccumulator{Float16}()
        kahan_add!(acc, Float16(1.0))
        kahan_add!(acc, Float16(0.5))
        @test acc.s == Float16(1.5)
    end
end
