# Birth-Death process model
# Reactions: ∅ → X (birth), X → ∅ (death)
# Features: Linear propensities, Poisson steady state

"""
    BirthDeathModel

Birth-Death process with linear propensities.

# Fields
- `birth_rate::Float64`: Rate of birth reaction (∅ → X)
- `death_rate::Float64`: Rate of death reaction per molecule (X → ∅)
- `initial_population::Int`: Initial population count

# Steady-state properties
At equilibrium, population follows Poisson(λ) where λ = birth_rate / death_rate.
"""
Base.@kwdef struct BirthDeathModel
    birth_rate::Float64 = 1.0
    death_rate::Float64 = 0.1
    initial_population::Int = 10
end

"""
    compute_propensities(model::BirthDeathModel, n::Integer, ::Type{T}) where T

Compute propensities for birth and death reactions.

Returns (a_birth, a_death) as type T.
- Birth: a₁ = birth_rate (constant, zeroth order)
- Death: a₂ = death_rate * n (first order in population)
"""
function compute_propensities(model::BirthDeathModel, n::Integer, ::Type{T}) where T
    a_birth = T(model.birth_rate)
    a_death = T(model.death_rate) * T(n)
    return (a_birth, a_death)
end

"""
    apply_reaction(n::Integer, reaction_idx::Integer) -> Int

Apply the selected reaction to the population.
- reaction_idx == 1: Birth (n → n+1)
- reaction_idx == 2: Death (n → n-1)
"""
function apply_reaction(n::Integer, reaction_idx::Integer)
    if reaction_idx == 1
        return n + 1
    else
        return max(0, n - 1)  # Prevent negative populations
    end
end

"""
    ssa_birth_death!(model::BirthDeathModel, t_end::Real;
                     Tprop::Type=Float64,
                     Tacc::Type=Float32,
                     rng::AbstractRNG=Random.default_rng(),
                     mode::Symbol=:mixed,
                     kahan::Bool=false)

Run the Stochastic Simulation Algorithm (Direct Method) for the Birth-Death process.

# Arguments
- `model`: BirthDeathModel specification
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
- `counts::Vector{Int}`: Population counts after each event
- `n_events::Int`: Total number of events
- `model::BirthDeathModel`: Model specification used
- `cfg::NamedTuple`: Configuration (Tprop, Tacc, mode, kahan)
"""
function ssa_birth_death!(model::BirthDeathModel, t_end::Real;
                          Tprop::Type=Float64,
                          Tacc::Type=Float32,
                          rng::AbstractRNG=Random.default_rng(),
                          mode::Symbol=:mixed,
                          kahan::Bool=false)
    if mode === :strict
        return _ssa_birth_death_strict!(model, t_end; Tprop, rng, kahan)
    else
        return _ssa_birth_death_mixed!(model, t_end; Tprop, Tacc, rng)
    end
end

function _ssa_birth_death_mixed!(model::BirthDeathModel, t_end::Real;
                                  Tprop::Type=Float64,
                                  Tacc::Type=Float32,
                                  rng::AbstractRNG=Random.default_rng())
    n = model.initial_population
    t = 0.0

    times = Float64[0.0]
    counts = Int[n]

    n_events = 0
    while t < t_end
        a_birth, a_death = compute_propensities(model, n, Tprop)
        a0 = Tacc(a_birth) + Tacc(a_death)

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
        if Float32(a_birth) >= threshold
            reaction_idx = 1
        else
            reaction_idx = 2
        end

        n = apply_reaction(n, reaction_idx)
        n_events += 1
        push!(times, t)
        push!(counts, n)
    end

    return (times=times, counts=counts, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tacc, mode=:mixed, kahan=false))
end

function _ssa_birth_death_strict!(model::BirthDeathModel, t_end::Real;
                                   Tprop::Type=Float64,
                                   rng::AbstractRNG=Random.default_rng(),
                                   kahan::Bool=false)
    n = model.initial_population
    t = zero(Tprop)
    t_end_T = Tprop(t_end)

    times = Float64[0.0]
    counts = Int[n]

    kacc = kahan ? KahanAccumulator{Tprop}() : nothing

    n_events = 0
    while t < t_end_T
        a_birth, a_death = compute_propensities(model, n, Tprop)
        a0 = a_birth + a_death

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
        if a_birth >= threshold
            reaction_idx = 1
        else
            reaction_idx = 2
        end

        n = apply_reaction(n, reaction_idx)
        n_events += 1
        push!(times, Float64(t))
        push!(counts, n)
    end

    return (times=times, counts=counts, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tprop, mode=:strict, kahan=kahan))
end
