# Tests for statistical analysis utilities

using Test
using LowPrecBio
using Distributions
using Random

@testset "Statistical utilities" begin
    @testset "Wasserstein distance" begin
        # Known value: all-zeros vs all-ones should give distance 1.0
        @test wasserstein_distance(zeros(100), ones(100)) ≈ 1.0

        # Self-distance is zero
        data = randn(MersenneTwister(42), 200)
        @test wasserstein_distance(data, data) ≈ 0.0 atol=1e-12

        # Symmetry
        a = randn(MersenneTwister(1), 100)
        b = randn(MersenneTwister(2), 100)
        @test wasserstein_distance(a, b) ≈ wasserstein_distance(b, a)
    end

    @testset "KS two-sample test" begin
        rng = MersenneTwister(123)
        s1 = randn(rng, 500)
        s2 = randn(rng, 500)
        result = ks_two_sample_test(s1, s2)

        # Returns named tuple with correct fields
        @test haskey(result, :statistic)
        @test haskey(result, :p_value)

        # Same-distribution samples should not reject at α=0.05
        @test result.p_value > 0.05
    end

    @testset "Mean with confidence interval" begin
        rng = MersenneTwister(456)
        data = 5.0 .+ randn(rng, 10_000)  # true mean = 5
        result = mean_with_ci(data)

        # Mean should be close to true value
        @test abs(result.mean - 5.0) < 0.1

        # CI should contain the true mean
        @test result.lower < 5.0 < result.upper

        # Lower < mean < upper
        @test result.lower < result.mean < result.upper
    end

    @testset "Variance with confidence interval" begin
        rng = MersenneTwister(789)
        data = 3.0 * randn(rng, 10_000)  # true variance = 9
        result = variance_with_ci(data)

        # Variance should be close to true value
        @test abs(result.variance - 9.0) < 1.0

        # CI should contain the true variance
        @test result.lower < 9.0 < result.upper

        # Lower < variance < upper
        @test result.lower < result.variance < result.upper
    end

    @testset "Negative population count" begin
        @test negative_population_count([1, 2, 3, -1, -2]) == 2
        @test negative_population_count([0, 1, 2]) == 0
        @test negative_population_count([-1, -2, -3]) == 3
    end

    @testset "Underflow/overflow counts" begin
        small = Float64(floatmin(Float16)) / 2   # underflows in Float16
        big   = Float64(floatmax(Float16)) * 2   # overflows in Float16
        vals  = [0.0, 1.0, small, big, -big]
        result = underflow_overflow_counts(vals, Float16)
        @test result.n_underflow == 1
        @test result.n_overflow == 2
    end

    @testset "Reaction channel frequencies" begin
        events = [1, 1, 2, 3, 3, 3]
        freqs = reaction_channel_frequencies(events, 3)
        @test length(freqs) == 3
        @test freqs ≈ [2/6, 1/6, 3/6]
        @test sum(freqs) ≈ 1.0

        # Empty events
        @test all(reaction_channel_frequencies(Int[], 3) .== 0.0)
    end
end

@testset "Dwell time analysis" begin
    @testset "Dwell time extraction" begin
        # Synthetic trajectory: 0,0 -> 1,1,1 -> 0,0 -> 1
        states = [0, 0, 1, 1, 1, 0, 0, 1]
        times  = [0, 1, 2, 3, 4, 5, 6, 7]
        result = extract_dwell_times(states, times)

        # off: [0,1]->[2] = 2.0, [5,6]->[7] = 2.0 (but second off ends at transition to 1 at t=7)
        # on: [2,3,4]->[5] = 3.0
        # Final segment: state=1 at t=7, dwell from t=7 to t=7 = 0.0
        @test length(result.off_dwells) == 2
        @test result.off_dwells[1] ≈ 2.0
        @test result.off_dwells[2] ≈ 2.0
        @test length(result.on_dwells) == 2
        @test result.on_dwells[1] ≈ 3.0
        @test result.on_dwells[2] ≈ 0.0  # final segment from t=7 to t=7

        # Edge case: constant state → one dwell covering full span
        states_const = [1, 1, 1, 1]
        times_const  = [0, 1, 2, 3]
        rc = extract_dwell_times(states_const, times_const)
        @test length(rc.on_dwells) == 1
        @test rc.on_dwells[1] ≈ 3.0
        @test isempty(rc.off_dwells)
    end

    @testset "Exponential fitting" begin
        rng = MersenneTwister(42)
        true_rate = 0.5
        data = rand(rng, Exponential(1 / true_rate), 10_000)
        result = fit_exponential(data)

        # Rate should be close to true value
        @test abs(result.rate - true_rate) < 0.05
        @test result.n == 10_000

        # CI should contain the true rate
        @test result.lower < true_rate < result.upper

        # Empty input returns NaN
        empty_result = fit_exponential(Float64[])
        @test isnan(empty_result.rate)
        @test isnan(empty_result.lower)
        @test isnan(empty_result.upper)
        @test empty_result.n == 0
    end

    @testset "Well occupancy" begin
        counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = well_occupancy(counts, 55)

        @test result.n_low == 5
        @test result.n_high == 5
        @test result.frac_low ≈ 0.5
        @test result.frac_high ≈ 0.5
        @test result.frac_low + result.frac_high ≈ 1.0

        # All below threshold
        r2 = well_occupancy([1, 2, 3], 10)
        @test r2.n_low == 3
        @test r2.n_high == 0
        @test r2.frac_low ≈ 1.0

        # Empty input
        r3 = well_occupancy(Int[], 5)
        @test isnan(r3.frac_low)
    end

    @testset "Mean first passage time" begin
        # Synthetic trajectory crossing threshold=5 at known times
        counts = [2, 3, 6, 8, 4, 2, 7, 9, 3, 1, 6]
        times  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = mean_first_passage_time(counts, times, 5)

        @test result.n_crossings > 0
        @test result.mfpt > 0

        # No crossings
        r2 = mean_first_passage_time([1, 2, 3], [0, 1, 2], 10)
        @test isnan(r2.mfpt)
        @test r2.n_crossings == 0
    end

    @testset "Distribution tests" begin
        # Anderson-Darling test with exponential data
        rng = MersenneTwister(101)
        data = rand(rng, Exponential(2.0), 500)
        result = anderson_darling_test(data, Exponential(2.0))
        @test haskey(result, :statistic)
        @test haskey(result, :p_value)
        @test result.p_value > 0.05  # should not reject correct distribution
    end
end
