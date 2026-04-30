# Repressilator model (3-gene cyclic repression network)
# Species: mA, mB, mC (mRNAs), pA, pB, pC (proteins)
# Reactions (4 per gene, 12 total):
#   Gene A (repressed by protein C):
#     R1:  ∅  → mA   (transcription, rate = alpha0 + alpha / (1 + pC^n))
#     R2:  mA → ∅    (mRNA degradation, rate = delta_m * mA)
#     R3:  ∅  → pA   (translation, rate = beta * mA)
#     R4:  pA → ∅    (protein degradation, rate = delta_p * pA)
#   Gene B (repressed by protein A):
#     R5:  ∅  → mB   (transcription, rate = alpha0 + alpha / (1 + pA^n))
#     R6:  mB → ∅    (mRNA degradation, rate = delta_m * mB)
#     R7:  ∅  → pB   (translation, rate = beta * mB)
#     R8:  pB → ∅    (protein degradation, rate = delta_p * pB)
#   Gene C (repressed by protein B):
#     R9:  ∅  → mC   (transcription, rate = alpha0 + alpha / (1 + pB^n))
#     R10: mC → ∅    (mRNA degradation, rate = delta_m * mC)
#     R11: ∅  → pC   (translation, rate = beta * mC)
#     R12: pC → ∅    (protein degradation, rate = delta_p * pC)

"""
    RepressilatorModel

Repressilator: 3-gene cyclic repression network producing sustained oscillations.

# Fields
- `alpha0::Float64`: Basal transcription rate (leakiness)
- `alpha::Float64`: Maximum regulated transcription rate
- `n::Int`: Hill coefficient for repression
- `delta_m::Float64`: mRNA degradation rate
- `beta::Float64`: Translation rate (protein synthesis per mRNA)
- `delta_p::Float64`: Protein degradation rate
- `initial_mA::Int`: Initial mRNA count for gene A
- `initial_mB::Int`: Initial mRNA count for gene B
- `initial_mC::Int`: Initial mRNA count for gene C
- `initial_pA::Int`: Initial protein count for gene A
- `initial_pB::Int`: Initial protein count for gene B
- `initial_pC::Int`: Initial protein count for gene C

# Dynamics
Each gene represses the next in a cycle: A ⊣ B ⊣ C ⊣ A.
Transcription uses a Hill function: rate = alpha0 + alpha / (1 + repressor^n).
This produces sustained oscillations when parameters are appropriate.
"""
Base.@kwdef struct RepressilatorModel
    alpha0::Float64 = 1.0      # basal transcription rate
    alpha::Float64 = 216.0     # max regulated transcription rate
    n::Int = 2                 # Hill coefficient
    delta_m::Float64 = 1.0     # mRNA degradation rate
    beta::Float64 = 5.0        # translation rate
    delta_p::Float64 = 1.0     # protein degradation rate
    initial_mA::Int = 0
    initial_mB::Int = 0
    initial_mC::Int = 0
    initial_pA::Int = 5        # asymmetric to break symmetry
    initial_pB::Int = 0
    initial_pC::Int = 0
end

"""
    compute_propensities(model::RepressilatorModel, mA::Integer, mB::Integer, mC::Integer,
                         pA::Integer, pB::Integer, pC::Integer, ::Type{T}) where T

Compute propensities for all 12 Repressilator reactions.

Returns a tuple of 12 propensities as type T.
Hill repression: rate = alpha0 + alpha / (1 + repressor^n).
"""
function compute_propensities(model::RepressilatorModel,
                              mA::Integer, mB::Integer, mC::Integer,
                              pA::Integer, pB::Integer, pC::Integer,
                              ::Type{T}) where T
    # Hill repression functions (computed in Tprop)
    pC_n = T(pC)^model.n
    pA_n = T(pA)^model.n
    pB_n = T(pB)^model.n

    hill_A = T(model.alpha0) + T(model.alpha) / (T(1) + pC_n)  # gene A repressed by pC
    hill_B = T(model.alpha0) + T(model.alpha) / (T(1) + pA_n)  # gene B repressed by pA
    hill_C = T(model.alpha0) + T(model.alpha) / (T(1) + pB_n)  # gene C repressed by pB

    # Gene A reactions
    a1  = hill_A                            # mA synthesis
    a2  = T(model.delta_m) * T(mA)         # mA degradation
    a3  = T(model.beta) * T(mA)            # pA synthesis
    a4  = T(model.delta_p) * T(pA)         # pA degradation

    # Gene B reactions
    a5  = hill_B                            # mB synthesis
    a6  = T(model.delta_m) * T(mB)         # mB degradation
    a7  = T(model.beta) * T(mB)            # pB synthesis
    a8  = T(model.delta_p) * T(pB)         # pB degradation

    # Gene C reactions
    a9  = hill_C                            # mC synthesis
    a10 = T(model.delta_m) * T(mC)         # mC degradation
    a11 = T(model.beta) * T(mC)            # pC synthesis
    a12 = T(model.delta_p) * T(pC)         # pC degradation

    return (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
end

"""
    apply_reaction(model::RepressilatorModel, mA::Integer, mB::Integer, mC::Integer,
                   pA::Integer, pB::Integer, pC::Integer, reaction_idx::Integer)

Apply the selected reaction to the state (mA, mB, mC, pA, pB, pC).

Returns updated (mA, mB, mC, pA, pB, pC) tuple.
"""
function apply_reaction(model::RepressilatorModel,
                        mA::Integer, mB::Integer, mC::Integer,
                        pA::Integer, pB::Integer, pC::Integer,
                        reaction_idx::Integer)
    if reaction_idx == 1
        return (mA + 1, mB, mC, pA, pB, pC)            # mA synthesis
    elseif reaction_idx == 2
        return (max(0, mA - 1), mB, mC, pA, pB, pC)    # mA degradation
    elseif reaction_idx == 3
        return (mA, mB, mC, pA + 1, pB, pC)             # pA synthesis
    elseif reaction_idx == 4
        return (mA, mB, mC, max(0, pA - 1), pB, pC)     # pA degradation
    elseif reaction_idx == 5
        return (mA, mB + 1, mC, pA, pB, pC)             # mB synthesis
    elseif reaction_idx == 6
        return (mA, max(0, mB - 1), mC, pA, pB, pC)     # mB degradation
    elseif reaction_idx == 7
        return (mA, mB, mC, pA, pB + 1, pC)             # pB synthesis
    elseif reaction_idx == 8
        return (mA, mB, mC, pA, max(0, pB - 1), pC)     # pB degradation
    elseif reaction_idx == 9
        return (mA, mB, mC + 1, pA, pB, pC)             # mC synthesis
    elseif reaction_idx == 10
        return (mA, mB, max(0, mC - 1), pA, pB, pC)     # mC degradation
    elseif reaction_idx == 11
        return (mA, mB, mC, pA, pB, pC + 1)             # pC synthesis
    else
        return (mA, mB, mC, pA, pB, max(0, pC - 1))     # pC degradation
    end
end

"""
    ssa_repressilator!(model::RepressilatorModel, t_end::Real;
                       Tprop::Type=Float64,
                       Tacc::Type=Float32,
                       rng::AbstractRNG=Random.default_rng(),
                       mode::Symbol=:mixed,
                       kahan::Bool=false)

Run the Stochastic Simulation Algorithm (Direct Method) for the Repressilator model.

# Arguments
- `model`: RepressilatorModel specification
- `t_end`: Simulation end time

# Keyword Arguments
- `Tprop`: Type for propensity calculations (default: Float64)
- `Tacc`: Type for accumulation operations (default: Float32)
- `rng`: Random number generator (default: default_rng())
- `mode`: `:mixed` (default) uses Tacc for accumulation; `:strict` uses Tprop throughout
- `kahan`: If `true` and `mode=:strict`, use Kahan compensated summation for time

# Returns
Named tuple with:
- `times::Vector{Float64}`: Event times
- `counts_mA::Vector{Int}`: mRNA A counts after each event
- `counts_mB::Vector{Int}`: mRNA B counts after each event
- `counts_mC::Vector{Int}`: mRNA C counts after each event
- `counts_pA::Vector{Int}`: Protein A counts after each event
- `counts_pB::Vector{Int}`: Protein B counts after each event
- `counts_pC::Vector{Int}`: Protein C counts after each event
- `n_events::Int`: Total number of events
- `model::RepressilatorModel`: Model specification used
- `cfg::NamedTuple`: Configuration (Tprop, Tacc, mode, kahan)
"""
function ssa_repressilator!(model::RepressilatorModel, t_end::Real;
                            Tprop::Type=Float64,
                            Tacc::Type=Float32,
                            rng::AbstractRNG=Random.default_rng(),
                            mode::Symbol=:mixed,
                            kahan::Bool=false,
                            max_events::Int=typemax(Int))
    if mode === :strict
        return _ssa_repressilator_strict!(model, t_end; Tprop, rng, kahan, max_events)
    else
        return _ssa_repressilator_mixed!(model, t_end; Tprop, Tacc, rng)
    end
end

function _ssa_repressilator_mixed!(model::RepressilatorModel, t_end::Real;
                                    Tprop::Type=Float64,
                                    Tacc::Type=Float32,
                                    rng::AbstractRNG=Random.default_rng())
    mA = model.initial_mA
    mB = model.initial_mB
    mC = model.initial_mC
    pA = model.initial_pA
    pB = model.initial_pB
    pC = model.initial_pC
    t = 0.0

    times = Float64[0.0]
    counts_mA = Int[mA]
    counts_mB = Int[mB]
    counts_mC = Int[mC]
    counts_pA = Int[pA]
    counts_pB = Int[pB]
    counts_pC = Int[pC]

    n_events = 0
    while t < t_end
        props = compute_propensities(model, mA, mB, mC, pA, pB, pC, Tprop)

        a0 = Tacc(0)
        for ai in props
            a0 += Tacc(ai)
        end

        if a0 <= zero(Tacc)
            break
        end

        r1 = rand(rng, Float32)
        r2 = rand(rng, Float32)
        τ = -log(r1) / Float32(a0)

        t_new = t + Float64(τ)
        if t_new > t_end
            break
        end
        t = t_new

        threshold = r2 * Float32(a0)
        cumsum = Float32(0)
        reaction_idx = 12
        for i in 1:12
            cumsum += Float32(props[i])
            if cumsum >= threshold
                reaction_idx = i
                break
            end
        end

        mA, mB, mC, pA, pB, pC = apply_reaction(model, mA, mB, mC, pA, pB, pC, reaction_idx)
        n_events += 1

        push!(times, t)
        push!(counts_mA, mA)
        push!(counts_mB, mB)
        push!(counts_mC, mC)
        push!(counts_pA, pA)
        push!(counts_pB, pB)
        push!(counts_pC, pC)
    end

    return (times=times, counts_mA=counts_mA, counts_mB=counts_mB, counts_mC=counts_mC,
            counts_pA=counts_pA, counts_pB=counts_pB, counts_pC=counts_pC,
            n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tacc, mode=:mixed, kahan=false))
end

function _ssa_repressilator_strict!(model::RepressilatorModel, t_end::Real;
                                     Tprop::Type=Float64,
                                     rng::AbstractRNG=Random.default_rng(),
                                     kahan::Bool=false,
                                     max_events::Int=typemax(Int))
    mA = model.initial_mA
    mB = model.initial_mB
    mC = model.initial_mC
    pA = model.initial_pA
    pB = model.initial_pB
    pC = model.initial_pC
    t = zero(Tprop)
    t_end_T = Tprop(t_end)

    times = Float64[0.0]
    counts_mA = Int[mA]
    counts_mB = Int[mB]
    counts_mC = Int[mC]
    counts_pA = Int[pA]
    counts_pB = Int[pB]
    counts_pC = Int[pC]

    kacc = kahan ? KahanAccumulator{Tprop}() : nothing

    n_events = 0
    while t < t_end_T && n_events < max_events
        props = compute_propensities(model, mA, mB, mC, pA, pB, pC, Tprop)

        a0 = zero(Tprop)
        for ai in props
            a0 += ai
        end

        if a0 <= zero(Tprop)
            break
        end

        u1 = clamp(Tprop(rand(rng, Float64)), nextfloat(zero(Tprop)), prevfloat(one(Tprop)))
        u2 = Tprop(rand(rng, Float64))
        τ = -log(u1) / a0

        if kahan
            kahan_add!(kacc, τ)
            t = kacc.s
        else
            t = t + τ
        end

        if t > t_end_T
            break
        end

        threshold = u2 * a0
        cs = zero(Tprop)
        reaction_idx = 12
        for i in 1:12
            cs += props[i]
            if cs >= threshold
                reaction_idx = i
                break
            end
        end

        mA, mB, mC, pA, pB, pC = apply_reaction(model, mA, mB, mC, pA, pB, pC, reaction_idx)
        n_events += 1

        push!(times, Float64(t))
        push!(counts_mA, mA)
        push!(counts_mB, mB)
        push!(counts_mC, mC)
        push!(counts_pA, pA)
        push!(counts_pB, pB)
        push!(counts_pC, pC)
    end

    return (times=times, counts_mA=counts_mA, counts_mB=counts_mB, counts_mC=counts_mC,
            counts_pA=counts_pA, counts_pB=counts_pB, counts_pC=counts_pC,
            n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tprop, mode=:strict, kahan=kahan))
end
