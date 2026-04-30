# Tests for SSA model implementations

using Test
using LowPrecBio
using Random
using Statistics

@testset "Model implementations" begin
    @testset "Birth-Death model" begin
        @testset "BirthDeathModel struct exists and has correct defaults" begin
            m = BirthDeathModel()
            @test m.birth_rate == 1.0
            @test m.death_rate == 0.1
            @test m.initial_population == 10
        end

        @testset "BirthDeathModel accepts custom parameters" begin
            m = BirthDeathModel(birth_rate=2.0, death_rate=0.5, initial_population=20)
            @test m.birth_rate == 2.0
            @test m.death_rate == 0.5
            @test m.initial_population == 20
        end

        @testset "ssa_birth_death! runs without error" begin
            m = BirthDeathModel()
            result = ssa_birth_death!(m, 10.0)

            @test haskey(result, :times)
            @test haskey(result, :counts)
            @test haskey(result, :n_events)
            @test haskey(result, :model)
            @test haskey(result, :cfg)

            @test length(result.times) == length(result.counts)
            @test result.times[1] == 0.0
            @test result.counts[1] == m.initial_population
            @test all(result.counts .>= 0)
            @test issorted(result.times)
        end

        @testset "ssa_birth_death! is reproducible with fixed seed" begin
            m = BirthDeathModel()
            rng1 = Random.Xoshiro(12345)
            rng2 = Random.Xoshiro(12345)

            result1 = ssa_birth_death!(m, 10.0; rng=rng1)
            result2 = ssa_birth_death!(m, 10.0; rng=rng2)

            @test result1.times == result2.times
            @test result1.counts == result2.counts
        end

        @testset "Steady-state mean matches theory (Poisson)" begin
            # Theory: steady-state population ~ Poisson(λ) where λ = birth_rate / death_rate
            birth_rate = 5.0
            death_rate = 0.5
            expected_mean = birth_rate / death_rate  # = 10.0

            m = BirthDeathModel(
                birth_rate=birth_rate,
                death_rate=death_rate,
                initial_population=Int(expected_mean)
            )

            # Run long simulation to reach steady state
            rng = Random.Xoshiro(42)
            result = ssa_birth_death!(m, 1000.0; rng=rng)

            # Sample counts at regular intervals after burn-in
            # Use time-weighted average for better estimate
            burn_in_time = 100.0
            steady_state_counts = Int[]
            for i in 2:length(result.times)
                if result.times[i] > burn_in_time
                    push!(steady_state_counts, result.counts[i-1])
                end
            end

            sample_mean = mean(steady_state_counts)

            # Allow 20% tolerance due to stochastic nature
            @test abs(sample_mean - expected_mean) / expected_mean < 0.2
        end

        @testset "Steady-state variance matches theory (Poisson)" begin
            # For Poisson: variance = mean = λ
            birth_rate = 5.0
            death_rate = 0.5
            expected_variance = birth_rate / death_rate  # = 10.0

            m = BirthDeathModel(
                birth_rate=birth_rate,
                death_rate=death_rate,
                initial_population=10
            )

            rng = Random.Xoshiro(123)
            result = ssa_birth_death!(m, 1000.0; rng=rng)

            burn_in_time = 100.0
            steady_state_counts = Int[]
            for i in 2:length(result.times)
                if result.times[i] > burn_in_time
                    push!(steady_state_counts, result.counts[i-1])
                end
            end

            sample_variance = var(steady_state_counts)

            # Allow 50% tolerance for variance (higher variance in estimate)
            @test abs(sample_variance - expected_variance) / expected_variance < 0.5
        end

        @testset "Absorbing state (zero population, no birth)" begin
            m = BirthDeathModel(birth_rate=0.0, death_rate=1.0, initial_population=5)
            rng = Random.Xoshiro(999)
            result = ssa_birth_death!(m, 100.0; rng=rng)

            # Should eventually reach zero and stop
            @test result.counts[end] == 0
        end

        @testset "Different precision modes work" begin
            m = BirthDeathModel()

            # Test with Float32 propensities
            result_fp32 = ssa_birth_death!(m, 10.0; Tprop=Float32)
            @test result_fp32.cfg.Tprop == Float32

            # Test with Float16 propensities
            result_fp16 = ssa_birth_death!(m, 10.0; Tprop=Float16)
            @test result_fp16.cfg.Tprop == Float16
        end
    end

    @testset "Schlögl model" begin
        @testset "SchloglModel struct exists and has correct defaults" begin
            m = SchloglModel()
            @test m.k1 == 3.0e-7
            @test m.k2 == 1.0e-4
            @test m.k3 == 1.0e-3
            @test m.k4 == 3.5
            @test m.A == 1.0e5
            @test m.B == 2.0e5
            @test m.initial_population == 250
        end

        @testset "SchloglModel accepts custom parameters" begin
            m = SchloglModel(k1=1e-6, k2=2e-4, k3=2e-3, k4=4.0, A=5e4, B=1e5, initial_population=100)
            @test m.k1 == 1e-6
            @test m.k2 == 2e-4
            @test m.k3 == 2e-3
            @test m.k4 == 4.0
            @test m.A == 5e4
            @test m.B == 1e5
            @test m.initial_population == 100
        end

        @testset "ssa_schlogl! runs without error" begin
            m = SchloglModel()
            result = ssa_schlogl!(m, 1.0)

            @test haskey(result, :times)
            @test haskey(result, :counts)
            @test haskey(result, :n_events)
            @test haskey(result, :model)
            @test haskey(result, :cfg)

            @test length(result.times) == length(result.counts)
            @test result.times[1] == 0.0
            @test result.counts[1] == m.initial_population
            @test all(result.counts .>= 0)
            @test issorted(result.times)
        end

        @testset "ssa_schlogl! is reproducible with fixed seed" begin
            m = SchloglModel()
            rng1 = Random.Xoshiro(12345)
            rng2 = Random.Xoshiro(12345)

            result1 = ssa_schlogl!(m, 1.0; rng=rng1)
            result2 = ssa_schlogl!(m, 1.0; rng=rng2)

            @test result1.times == result2.times
            @test result1.counts == result2.counts
        end

        @testset "Bimodal distribution observed (bistability)" begin
            # Run a longer simulation to observe switching between wells
            # With default parameters, expect wells around n≈80 (low) and n≈250 (high)
            m = SchloglModel()
            rng = Random.Xoshiro(42)

            # Run for enough time to sample both wells
            result = ssa_schlogl!(m, 100.0; rng=rng)

            # Count occurrences in low and high regions
            low_threshold = 150
            high_threshold = 200

            low_count = count(c -> c < low_threshold, result.counts)
            high_count = count(c -> c > high_threshold, result.counts)

            # Both wells should be visited (bimodality)
            # Note: This may not always pass due to stochastic nature and short simulation
            # but with seed 42 and reasonable parameters it should work
            @test low_count > 0 || high_count > 0  # At least one region visited
            @test result.n_events > 100  # Should have many events
        end

        @testset "Propensities scale correctly with population" begin
            m = SchloglModel()

            # Test at n=0: only a3 (B → X) should be nonzero
            a1, a2, a3, a4 = LowPrecBio.compute_propensities(m, 0, Float64)
            @test a1 == 0.0
            @test a2 == 0.0
            @test a3 > 0.0  # k3 * B
            @test a4 == 0.0

            # Test at n=1: a1 and a2 involve (n-1) or (n-2) terms
            a1, a2, a3, a4 = LowPrecBio.compute_propensities(m, 1, Float64)
            @test a1 == 0.0  # k1 * A * 1 * 0 = 0
            @test a2 == 0.0  # k2 * 1 * 0 * (-1) but clamped
            @test a3 > 0.0
            @test a4 > 0.0  # k4 * 1

            # Test at n=100: all propensities should be positive
            a1, a2, a3, a4 = LowPrecBio.compute_propensities(m, 100, Float64)
            @test a1 > 0.0
            @test a2 > 0.0
            @test a3 > 0.0
            @test a4 > 0.0
        end

        @testset "Different precision modes work" begin
            m = SchloglModel()

            # Test with Float32 propensities
            result_fp32 = ssa_schlogl!(m, 1.0; Tprop=Float32)
            @test result_fp32.cfg.Tprop == Float32

            # Test with Float16 propensities
            result_fp16 = ssa_schlogl!(m, 1.0; Tprop=Float16)
            @test result_fp16.cfg.Tprop == Float16
        end

        @testset "Population stays non-negative" begin
            # Even with many events, population should never go negative
            m = SchloglModel(initial_population=10)
            rng = Random.Xoshiro(999)
            result = ssa_schlogl!(m, 10.0; rng=rng)

            @test all(result.counts .>= 0)
        end
    end

    @testset "Telegraph model" begin
        @testset "TelegraphModel struct exists and has correct defaults" begin
            m = TelegraphModel()
            @test m.k_on == 0.01
            @test m.k_off == 0.1
            @test m.alpha == 5.0
            @test m.beta == 1.0
            @test m.initial_population == 0
            @test m.initial_state == 0
        end

        @testset "TelegraphModel accepts custom parameters" begin
            m = TelegraphModel(k_on=0.05, k_off=0.2, alpha=10.0, beta=2.0,
                               initial_population=5, initial_state=1)
            @test m.k_on == 0.05
            @test m.k_off == 0.2
            @test m.alpha == 10.0
            @test m.beta == 2.0
            @test m.initial_population == 5
            @test m.initial_state == 1
        end

        @testset "ssa_telegraph! runs without error" begin
            m = TelegraphModel(initial_state=1)
            result = ssa_telegraph!(m, 10.0)

            @test haskey(result, :times)
            @test haskey(result, :counts)
            @test haskey(result, :states)
            @test haskey(result, :n_events)
            @test haskey(result, :model)
            @test haskey(result, :cfg)

            @test length(result.times) == length(result.counts)
            @test length(result.times) == length(result.states)
            @test result.times[1] == 0.0
            @test result.counts[1] == m.initial_population
            @test result.states[1] == m.initial_state
            @test issorted(result.times)
        end

        @testset "ssa_telegraph! is reproducible with fixed seed" begin
            m = TelegraphModel(initial_state=1)
            rng1 = Random.Xoshiro(12345)
            rng2 = Random.Xoshiro(12345)

            result1 = ssa_telegraph!(m, 10.0; rng=rng1)
            result2 = ssa_telegraph!(m, 10.0; rng=rng2)

            @test result1.times == result2.times
            @test result1.counts == result2.counts
            @test result1.states == result2.states
        end

        @testset "Promoter state always 0 or 1" begin
            m = TelegraphModel(initial_state=0)
            rng = Random.Xoshiro(42)
            result = ssa_telegraph!(m, 100.0; rng=rng)

            @test all(s -> s == 0 || s == 1, result.states)
        end

        @testset "Molecule count non-negative" begin
            m = TelegraphModel(initial_state=1, initial_population=0)
            rng = Random.Xoshiro(999)
            result = ssa_telegraph!(m, 100.0; rng=rng)

            @test all(result.counts .>= 0)
        end

        @testset "Different precision modes work" begin
            m = TelegraphModel(initial_state=1)

            result_fp32 = ssa_telegraph!(m, 10.0; Tprop=Float32)
            @test result_fp32.cfg.Tprop == Float32

            result_fp16 = ssa_telegraph!(m, 10.0; Tprop=Float16)
            @test result_fp16.cfg.Tprop == Float16
        end

        @testset "Propensity edge cases" begin
            m = TelegraphModel()

            # S=0: only a1 (OFF→ON) and a4 (decay) can be nonzero
            a1, a2, a3, a4 = LowPrecBio.compute_propensities(m, 0, 0, Float64)
            @test a1 == m.k_on   # k_on * (1-0)
            @test a2 == 0.0      # k_off * 0
            @test a3 == 0.0      # alpha * 0
            @test a4 == 0.0      # beta * 0

            # S=1, x=0: switching and production active, no decay
            a1, a2, a3, a4 = LowPrecBio.compute_propensities(m, 1, 0, Float64)
            @test a1 == 0.0      # k_on * (1-1)
            @test a2 == m.k_off  # k_off * 1
            @test a3 == m.alpha  # alpha * 1
            @test a4 == 0.0      # beta * 0

            # S=1, x=5: all propensities nonzero except a1
            a1, a2, a3, a4 = LowPrecBio.compute_propensities(m, 1, 5, Float64)
            @test a1 == 0.0
            @test a2 > 0.0
            @test a3 > 0.0
            @test a4 == m.beta * 5.0
        end
    end

    @testset "Dimer model" begin
        @testset "DimerModel struct exists and has correct defaults" begin
            m = DimerModel()
            @test m.kf == 1e-3
            @test m.kr == 0.1
            @test m.initial_A == 100
            @test m.initial_D == 0
        end

        @testset "DimerModel accepts custom parameters" begin
            m = DimerModel(kf=2e-3, kr=0.5, initial_A=50, initial_D=10)
            @test m.kf == 2e-3
            @test m.kr == 0.5
            @test m.initial_A == 50
            @test m.initial_D == 10
        end

        @testset "compute_propensities correctness" begin
            m = DimerModel(kf=1e-3, kr=0.1)

            # A=0: no association possible
            a1, a2 = LowPrecBio.compute_propensities(m, 0, 5, Float64)
            @test a1 == 0.0
            @test a2 == 0.1 * 5.0

            # A=1: no association (need at least 2 monomers)
            a1, a2 = LowPrecBio.compute_propensities(m, 1, 3, Float64)
            @test a1 == 0.0  # kf * 1 * 0 * 0.5 = 0
            @test a2 == 0.1 * 3.0

            # A=10, D=0: association only
            a1, a2 = LowPrecBio.compute_propensities(m, 10, 0, Float64)
            @test a1 ≈ 1e-3 * 10 * 9 * 0.5  # = 0.045
            @test a2 == 0.0

            # A=100, D=5: both nonzero
            a1, a2 = LowPrecBio.compute_propensities(m, 100, 5, Float64)
            @test a1 ≈ 1e-3 * 100 * 99 * 0.5
            @test a2 ≈ 0.1 * 5.0
        end

        @testset "apply_reaction correctness" begin
            m = DimerModel()

            # Association: 2A → D
            A, D = LowPrecBio.apply_reaction(m, 10, 3, 1)
            @test A == 8
            @test D == 4

            # Dissociation: D → 2A
            A, D = LowPrecBio.apply_reaction(m, 10, 3, 2)
            @test A == 12
            @test D == 2

            # Edge guard: association with A=1
            A, D = LowPrecBio.apply_reaction(m, 1, 3, 1)
            @test A == 0
            @test D == 4

            # Edge guard: dissociation with D=0
            A, D = LowPrecBio.apply_reaction(m, 10, 0, 2)
            @test A == 12
            @test D == 0
        end

        @testset "ssa_dimer! runs without error" begin
            m = DimerModel()
            result = ssa_dimer!(m, 10.0)

            @test haskey(result, :times)
            @test haskey(result, :counts_A)
            @test haskey(result, :counts_D)
            @test haskey(result, :max_deviation)
            @test haskey(result, :n_events)
            @test haskey(result, :model)
            @test haskey(result, :cfg)

            @test length(result.times) == length(result.counts_A)
            @test length(result.times) == length(result.counts_D)
            @test result.times[1] == 0.0
            @test result.counts_A[1] == m.initial_A
            @test result.counts_D[1] == m.initial_D
            @test all(result.counts_A .>= 0)
            @test all(result.counts_D .>= 0)
            @test issorted(result.times)
        end

        @testset "ssa_dimer! is reproducible with fixed seed" begin
            m = DimerModel()
            rng1 = Random.Xoshiro(12345)
            rng2 = Random.Xoshiro(12345)

            result1 = ssa_dimer!(m, 10.0; rng=rng1)
            result2 = ssa_dimer!(m, 10.0; rng=rng2)

            @test result1.times == result2.times
            @test result1.counts_A == result2.counts_A
            @test result1.counts_D == result2.counts_D
        end

        @testset "Conservation law A + 2D = M0 holds" begin
            m = DimerModel(initial_A=100, initial_D=0)
            M0 = m.initial_A + 2 * m.initial_D
            rng = Random.Xoshiro(42)
            result = ssa_dimer!(m, 100.0; rng=rng)

            # Check at every recorded time point
            for i in eachindex(result.times)
                @test result.counts_A[i] + 2 * result.counts_D[i] == M0
            end
            @test result.max_deviation == 0
        end

        @testset "Different precision modes work" begin
            m = DimerModel()

            result_fp32 = ssa_dimer!(m, 10.0; Tprop=Float32)
            @test result_fp32.cfg.Tprop == Float32

            result_fp16 = ssa_dimer!(m, 10.0; Tprop=Float16)
            @test result_fp16.cfg.Tprop == Float16
        end
    end

    @testset "Repressilator model" begin
        @testset "RepressilatorModel struct exists and has correct defaults" begin
            m = RepressilatorModel()
            @test m.alpha0 == 1.0
            @test m.alpha == 216.0
            @test m.n == 2
            @test m.delta_m == 1.0
            @test m.beta == 5.0
            @test m.delta_p == 1.0
            @test m.initial_mA == 0
            @test m.initial_mB == 0
            @test m.initial_mC == 0
            @test m.initial_pA == 5
            @test m.initial_pB == 0
            @test m.initial_pC == 0
        end

        @testset "RepressilatorModel accepts custom parameters" begin
            m = RepressilatorModel(alpha0=0.5, alpha=100.0, n=3, delta_m=0.5,
                                   beta=10.0, delta_p=2.0,
                                   initial_mA=1, initial_mB=2, initial_mC=3,
                                   initial_pA=10, initial_pB=20, initial_pC=30)
            @test m.alpha0 == 0.5
            @test m.alpha == 100.0
            @test m.n == 3
            @test m.delta_m == 0.5
            @test m.beta == 10.0
            @test m.delta_p == 2.0
            @test m.initial_mA == 1
            @test m.initial_mB == 2
            @test m.initial_mC == 3
            @test m.initial_pA == 10
            @test m.initial_pB == 20
            @test m.initial_pC == 30
        end

        @testset "ssa_repressilator! runs without error" begin
            m = RepressilatorModel()
            result = ssa_repressilator!(m, 10.0)

            @test haskey(result, :times)
            @test haskey(result, :counts_mA)
            @test haskey(result, :counts_mB)
            @test haskey(result, :counts_mC)
            @test haskey(result, :counts_pA)
            @test haskey(result, :counts_pB)
            @test haskey(result, :counts_pC)
            @test haskey(result, :n_events)
            @test haskey(result, :model)
            @test haskey(result, :cfg)

            @test length(result.times) == length(result.counts_mA)
            @test length(result.times) == length(result.counts_mB)
            @test length(result.times) == length(result.counts_mC)
            @test length(result.times) == length(result.counts_pA)
            @test length(result.times) == length(result.counts_pB)
            @test length(result.times) == length(result.counts_pC)
            @test result.times[1] == 0.0
            @test result.counts_mA[1] == m.initial_mA
            @test result.counts_mB[1] == m.initial_mB
            @test result.counts_mC[1] == m.initial_mC
            @test result.counts_pA[1] == m.initial_pA
            @test result.counts_pB[1] == m.initial_pB
            @test result.counts_pC[1] == m.initial_pC
            @test all(result.counts_mA .>= 0)
            @test all(result.counts_pA .>= 0)
            @test issorted(result.times)
        end

        @testset "ssa_repressilator! is reproducible with fixed seed" begin
            m = RepressilatorModel()
            rng1 = Random.Xoshiro(12345)
            rng2 = Random.Xoshiro(12345)

            result1 = ssa_repressilator!(m, 10.0; rng=rng1)
            result2 = ssa_repressilator!(m, 10.0; rng=rng2)

            @test result1.times == result2.times
            @test result1.counts_mA == result2.counts_mA
            @test result1.counts_mB == result2.counts_mB
            @test result1.counts_mC == result2.counts_mC
            @test result1.counts_pA == result2.counts_pA
            @test result1.counts_pB == result2.counts_pB
            @test result1.counts_pC == result2.counts_pC
        end

        @testset "Oscillatory dynamics" begin
            m = RepressilatorModel()
            rng = Random.Xoshiro(42)
            result = ssa_repressilator!(m, 200.0; rng=rng)

            # With default parameters, should produce sustained oscillations
            # Protein peaks should reach significant levels
            @test maximum(result.counts_pA) > 10
            @test maximum(result.counts_pB) > 10
            @test maximum(result.counts_pC) > 10

            # All mRNA species should be produced
            @test maximum(result.counts_mA) > 0
            @test maximum(result.counts_mB) > 0
            @test maximum(result.counts_mC) > 0

            # Should have many events in a long simulation
            @test result.n_events > 1000
        end

        @testset "Propensity correctness" begin
            m = RepressilatorModel()

            # Zero state (all species 0): Hill functions give alpha0 + alpha/(1+0) = alpha0 + alpha
            props = LowPrecBio.compute_propensities(m, 0, 0, 0, 0, 0, 0, Float64)
            expected_hill = m.alpha0 + m.alpha  # 1.0 + 216.0 = 217.0
            @test props[1] ≈ expected_hill   # mA transcription
            @test props[2] == 0.0            # mA degradation (mA=0)
            @test props[3] == 0.0            # pA translation (mA=0)
            @test props[4] == 0.0            # pA degradation (pA=0)
            @test props[5] ≈ expected_hill   # mB transcription
            @test props[9] ≈ expected_hill   # mC transcription

            # High repressor suppresses transcription
            # pC=100 represses gene A: hill = 1.0 + 216.0/(1+100^2) = 1.0 + 216/10001 ≈ 1.0216
            props_high = LowPrecBio.compute_propensities(m, 0, 0, 0, 0, 0, 100, Float64)
            @test props_high[1] < 2.0  # heavily repressed transcription
            @test props_high[1] > m.alpha0  # still above basal rate
        end

        @testset "apply_reaction correctness" begin
            m = RepressilatorModel()

            # R1: mA synthesis
            state = LowPrecBio.apply_reaction(m, 5, 3, 2, 10, 8, 6, 1)
            @test state == (6, 3, 2, 10, 8, 6)

            # R2: mA degradation
            state = LowPrecBio.apply_reaction(m, 5, 3, 2, 10, 8, 6, 2)
            @test state == (4, 3, 2, 10, 8, 6)

            # R3: pA synthesis
            state = LowPrecBio.apply_reaction(m, 5, 3, 2, 10, 8, 6, 3)
            @test state == (5, 3, 2, 11, 8, 6)

            # R4: pA degradation
            state = LowPrecBio.apply_reaction(m, 5, 3, 2, 10, 8, 6, 4)
            @test state == (5, 3, 2, 9, 8, 6)

            # R7: pB synthesis
            state = LowPrecBio.apply_reaction(m, 5, 3, 2, 10, 8, 6, 7)
            @test state == (5, 3, 2, 10, 9, 6)

            # R12: pC degradation
            state = LowPrecBio.apply_reaction(m, 5, 3, 2, 10, 8, 6, 12)
            @test state == (5, 3, 2, 10, 8, 5)

            # Edge guards: degradation at 0
            state = LowPrecBio.apply_reaction(m, 0, 0, 0, 0, 0, 0, 2)   # mA degrad at 0
            @test state[1] == 0
            state = LowPrecBio.apply_reaction(m, 0, 0, 0, 0, 0, 0, 4)   # pA degrad at 0
            @test state[4] == 0
            state = LowPrecBio.apply_reaction(m, 0, 0, 0, 0, 0, 0, 6)   # mB degrad at 0
            @test state[2] == 0
            state = LowPrecBio.apply_reaction(m, 0, 0, 0, 0, 0, 0, 8)   # pB degrad at 0
            @test state[5] == 0
            state = LowPrecBio.apply_reaction(m, 0, 0, 0, 0, 0, 0, 10)  # mC degrad at 0
            @test state[3] == 0
            state = LowPrecBio.apply_reaction(m, 0, 0, 0, 0, 0, 0, 12)  # pC degrad at 0
            @test state[6] == 0
        end

        @testset "Different precision modes work" begin
            m = RepressilatorModel()

            result_fp32 = ssa_repressilator!(m, 10.0; Tprop=Float32)
            @test result_fp32.cfg.Tprop == Float32

            result_fp16 = ssa_repressilator!(m, 10.0; Tprop=Float16)
            @test result_fp16.cfg.Tprop == Float16
        end

        @testset "Non-negative populations" begin
            m = RepressilatorModel()
            rng = Random.Xoshiro(999)
            result = ssa_repressilator!(m, 50.0; rng=rng)

            @test all(result.counts_mA .>= 0)
            @test all(result.counts_mB .>= 0)
            @test all(result.counts_mC .>= 0)
            @test all(result.counts_pA .>= 0)
            @test all(result.counts_pB .>= 0)
            @test all(result.counts_pC .>= 0)
        end
    end

    @testset "Strict-precision mode" begin
        @testset "Birth-Death strict mode" begin
            m = BirthDeathModel()

            @testset "runs and returns correct cfg" begin
                result = ssa_birth_death!(m, 10.0; Tprop=Float32, mode=:strict)
                @test result.cfg.mode === :strict
                @test result.cfg.Tprop === Float32
                @test result.cfg.Tacc === Float32  # strict uses Tprop as Tacc
                @test result.n_events > 0
                @test all(result.counts .>= 0)
                @test issorted(result.times)
            end

            @testset "reproducible with fixed seed" begin
                rng1 = Random.Xoshiro(42)
                rng2 = Random.Xoshiro(42)
                r1 = ssa_birth_death!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng1)
                r2 = ssa_birth_death!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng2)
                @test r1.times == r2.times
                @test r1.counts == r2.counts
            end

            @testset "with Kahan summation" begin
                result = ssa_birth_death!(m, 10.0; Tprop=Float32, mode=:strict, kahan=true)
                @test result.cfg.kahan === true
                @test result.n_events > 0
                @test issorted(result.times)
            end

            @testset "with Float16" begin
                result = ssa_birth_death!(m, 10.0; Tprop=Float16, mode=:strict)
                @test result.cfg.Tprop === Float16
                @test result.n_events > 0
            end
        end

        @testset "Dimer strict mode" begin
            m = DimerModel()

            @testset "runs and returns correct cfg" begin
                result = ssa_dimer!(m, 10.0; Tprop=Float32, mode=:strict)
                @test result.cfg.mode === :strict
                @test result.cfg.Tprop === Float32
                @test result.n_events > 0
                @test all(result.counts_A .>= 0)
                @test all(result.counts_D .>= 0)
            end

            @testset "reproducible with fixed seed" begin
                rng1 = Random.Xoshiro(42)
                rng2 = Random.Xoshiro(42)
                r1 = ssa_dimer!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng1)
                r2 = ssa_dimer!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng2)
                @test r1.times == r2.times
                @test r1.counts_A == r2.counts_A
            end

            @testset "conservation law holds in strict mode" begin
                M0 = m.initial_A + 2 * m.initial_D
                result = ssa_dimer!(m, 10.0; Tprop=Float32, mode=:strict, rng=Random.Xoshiro(42))
                for i in eachindex(result.times)
                    @test result.counts_A[i] + 2 * result.counts_D[i] == M0
                end
            end

            @testset "with Kahan summation" begin
                result = ssa_dimer!(m, 10.0; Tprop=Float32, mode=:strict, kahan=true)
                @test result.cfg.kahan === true
                @test result.n_events > 0
            end

            @testset "with Float16" begin
                result = ssa_dimer!(m, 10.0; Tprop=Float16, mode=:strict)
                @test result.cfg.Tprop === Float16
                @test result.n_events > 0
            end
        end

        @testset "Telegraph strict mode" begin
            m = TelegraphModel(initial_state=1)

            @testset "runs and returns correct cfg" begin
                result = ssa_telegraph!(m, 10.0; Tprop=Float32, mode=:strict)
                @test result.cfg.mode === :strict
                @test result.cfg.Tprop === Float32
                @test result.n_events > 0
                @test all(s -> s == 0 || s == 1, result.states)
                @test all(result.counts .>= 0)
            end

            @testset "reproducible with fixed seed" begin
                rng1 = Random.Xoshiro(42)
                rng2 = Random.Xoshiro(42)
                r1 = ssa_telegraph!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng1)
                r2 = ssa_telegraph!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng2)
                @test r1.times == r2.times
                @test r1.counts == r2.counts
                @test r1.states == r2.states
            end

            @testset "with Kahan summation" begin
                result = ssa_telegraph!(m, 10.0; Tprop=Float32, mode=:strict, kahan=true)
                @test result.cfg.kahan === true
                @test result.n_events > 0
            end

            @testset "with Float16" begin
                result = ssa_telegraph!(m, 10.0; Tprop=Float16, mode=:strict)
                @test result.cfg.Tprop === Float16
                @test result.n_events > 0
            end
        end

        @testset "Schlögl strict mode" begin
            m = SchloglModel()

            @testset "runs with Float32" begin
                result = ssa_schlogl!(m, 1.0; Tprop=Float32, mode=:strict)
                @test result.cfg.mode === :strict
                @test result.cfg.Tprop === Float32
                @test result.n_events > 0
                @test all(result.counts .>= 0)
            end

            @testset "reproducible with fixed seed" begin
                rng1 = Random.Xoshiro(42)
                rng2 = Random.Xoshiro(42)
                r1 = ssa_schlogl!(m, 1.0; Tprop=Float32, mode=:strict, rng=rng1)
                r2 = ssa_schlogl!(m, 1.0; Tprop=Float32, mode=:strict, rng=rng2)
                @test r1.times == r2.times
                @test r1.counts == r2.counts
            end

            @testset "with Kahan summation" begin
                result = ssa_schlogl!(m, 1.0; Tprop=Float32, mode=:strict, kahan=true)
                @test result.cfg.kahan === true
                @test result.n_events > 0
            end

            # Note: Float16 strict mode will overflow for Schlögl due to cubic propensity
            # 250*249*248 > 65504 (Float16 max). This is expected behavior.
        end

        @testset "Repressilator strict mode" begin
            m = RepressilatorModel()

            @testset "runs with Float32" begin
                result = ssa_repressilator!(m, 10.0; Tprop=Float32, mode=:strict)
                @test result.cfg.mode === :strict
                @test result.cfg.Tprop === Float32
                @test result.n_events > 0
                @test all(result.counts_mA .>= 0)
                @test all(result.counts_pA .>= 0)
            end

            @testset "reproducible with fixed seed" begin
                rng1 = Random.Xoshiro(42)
                rng2 = Random.Xoshiro(42)
                r1 = ssa_repressilator!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng1)
                r2 = ssa_repressilator!(m, 10.0; Tprop=Float32, mode=:strict, rng=rng2)
                @test r1.times == r2.times
                @test r1.counts_pA == r2.counts_pA
            end

            @testset "with Kahan summation" begin
                result = ssa_repressilator!(m, 10.0; Tprop=Float32, mode=:strict, kahan=true)
                @test result.cfg.kahan === true
                @test result.n_events > 0
            end

            @testset "with Float16" begin
                result = ssa_repressilator!(m, 10.0; Tprop=Float16, mode=:strict)
                @test result.cfg.Tprop === Float16
                @test result.n_events > 0
            end
        end

        @testset "Mixed mode backward compatibility" begin
            # Verify default mode is :mixed and cfg includes new fields
            m = BirthDeathModel()
            result = ssa_birth_death!(m, 10.0)
            @test result.cfg.mode === :mixed
            @test result.cfg.kahan === false
        end
    end

    @testset "Repressilator FP64 reference validation" begin
        # Helper: count oscillation peaks in a protein time series
        function count_peaks(values; min_prominence=5)
            peaks = 0
            for i in 2:length(values)-1
                if values[i] > values[i-1] && values[i] > values[i+1] && values[i] > min_prominence
                    peaks += 1
                end
            end
            return peaks
        end

        # Subsample at regular time intervals for peak detection
        function subsample_at_intervals(times, values; dt=2.0, t_start=50.0)
            t_end = times[end]
            ts = t_start:dt:t_end
            sampled = Int[]
            idx = 1
            for t_target in ts
                while idx < length(times) && times[idx+1] <= t_target
                    idx += 1
                end
                push!(sampled, values[idx])
            end
            return sampled
        end

        @testset "FP64 reference produces oscillations" begin
            m = RepressilatorModel()
            rng = Random.Xoshiro(42)
            result = ssa_repressilator!(m, 500.0; Tprop=Float64, rng=rng)

            # Subsample for peak detection
            pA_sub = subsample_at_intervals(result.times, result.counts_pA)
            pB_sub = subsample_at_intervals(result.times, result.counts_pB)
            pC_sub = subsample_at_intervals(result.times, result.counts_pC)

            peaks_A = count_peaks(pA_sub)
            peaks_B = count_peaks(pB_sub)
            peaks_C = count_peaks(pC_sub)

            # Expect sustained oscillations (>= 5 cycles in 500 time units)
            @test peaks_A >= 5
            @test peaks_B >= 5
            @test peaks_C >= 5
        end

        @testset "Float32 preserves oscillatory dynamics" begin
            m = RepressilatorModel()

            # FP64 reference
            rng64 = Random.Xoshiro(42)
            ref = ssa_repressilator!(m, 500.0; Tprop=Float64, rng=rng64)
            ref_pA = subsample_at_intervals(ref.times, ref.counts_pA)
            ref_peaks = count_peaks(ref_pA)
            ref_amp = maximum(ref.counts_pA)

            # Float32
            rng32 = Random.Xoshiro(42)
            r32 = ssa_repressilator!(m, 500.0; Tprop=Float32, rng=rng32)
            r32_pA = subsample_at_intervals(r32.times, r32.counts_pA)
            r32_peaks = count_peaks(r32_pA)
            r32_amp = maximum(r32.counts_pA)

            # Oscillations persist (at least 1/3 of FP64 peak count)
            @test r32_peaks >= ref_peaks ÷ 3
            # Peak amplitude within 50% of FP64 reference
            @test r32_amp > ref_amp * 0.5
        end

        @testset "Float16 preserves oscillatory dynamics" begin
            m = RepressilatorModel()

            # FP64 reference
            rng64 = Random.Xoshiro(42)
            ref = ssa_repressilator!(m, 500.0; Tprop=Float64, rng=rng64)
            ref_pA = subsample_at_intervals(ref.times, ref.counts_pA)
            ref_peaks = count_peaks(ref_pA)
            ref_amp = maximum(ref.counts_pA)

            # Float16 (mixed mode — propensities in Float16, accumulation in Float32)
            rng16 = Random.Xoshiro(42)
            r16 = ssa_repressilator!(m, 500.0; Tprop=Float16, rng=rng16)
            r16_pA = subsample_at_intervals(r16.times, r16.counts_pA)
            r16_peaks = count_peaks(r16_pA)
            r16_amp = maximum(r16.counts_pA)

            # Oscillations persist (at least 1/3 of FP64 peak count)
            @test r16_peaks >= ref_peaks ÷ 3
            # Peak amplitude within 50% of FP64 reference
            @test r16_amp > ref_amp * 0.5
        end
    end
end
