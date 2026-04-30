# Telegraph (gene switch) model
# Reactions:
#   R1: OFF → ON   (rate k_on,  propensity = k_on * (1-S))
#   R2: ON  → OFF  (rate k_off, propensity = k_off * S)
#   R3: ∅   → X    (rate alpha, propensity = alpha * S)   [production when ON]
#   R4: X   → ∅    (rate beta,  propensity = beta * x)    [decay]
# State variables: S ∈ {0,1} (promoter), x ∈ ℕ (molecule count)

"""
    TelegraphModel

Telegraph gene-switching model with bursty expression.

# Fields
- `k_on::Float64`: Rate of promoter activation (OFF → ON)
- `k_off::Float64`: Rate of promoter deactivation (ON → OFF)
- `alpha::Float64`: Production rate when promoter is ON
- `beta::Float64`: Decay rate per molecule
- `initial_population::Int`: Initial molecule count x₀
- `initial_state::Int`: Initial promoter state S₀ (0=OFF, 1=ON)

# Dynamics
The promoter switches stochastically between OFF (S=0) and ON (S=1).
Production only occurs in the ON state, leading to bursty gene expression.
"""
Base.@kwdef struct TelegraphModel
    k_on::Float64 = 0.01
    k_off::Float64 = 0.1
    alpha::Float64 = 5.0
    beta::Float64 = 1.0
    initial_population::Int = 0
    initial_state::Int = 0
end

"""
    compute_propensities(model::TelegraphModel, S::Integer, x::Integer, ::Type{T}) where T

Compute propensities for all four Telegraph reactions.

Returns (a1, a2, a3, a4) as type T:
- a1 = k_on * (1 - S)   (OFF → ON)
- a2 = k_off * S         (ON → OFF)
- a3 = alpha * S         (∅ → X, production)
- a4 = beta * x          (X → ∅, decay)
"""
function compute_propensities(model::TelegraphModel, S::Integer, x::Integer, ::Type{T}) where T
    a1 = T(model.k_on) * T(1 - S)
    a2 = T(model.k_off) * T(S)
    a3 = T(model.alpha) * T(S)
    a4 = T(model.beta) * T(x)
    return (a1, a2, a3, a4)
end

"""
    apply_reaction(model::TelegraphModel, S::Integer, x::Integer, reaction_idx::Integer) -> (Int, Int)

Apply the selected reaction to the state (S, x).
- reaction_idx == 1: OFF → ON  (S → 1)
- reaction_idx == 2: ON → OFF  (S → 0)
- reaction_idx == 3: ∅ → X     (x → x+1)
- reaction_idx == 4: X → ∅     (x → x-1)

Returns updated (S, x) tuple.
"""
function apply_reaction(model::TelegraphModel, S::Integer, x::Integer, reaction_idx::Integer)
    if reaction_idx == 1
        return (1, x)        # OFF → ON
    elseif reaction_idx == 2
        return (0, x)        # ON → OFF
    elseif reaction_idx == 3
        return (S, x + 1)   # ∅ → X (production)
    else
        return (S, max(0, x - 1))  # X → ∅ (decay)
    end
end

"""
    ssa_telegraph!(model::TelegraphModel, t_end::Real;
                   Tprop::Type=Float64,
                   Tacc::Type=Float32,
                   rng::AbstractRNG=Random.default_rng(),
                   mode::Symbol=:mixed,
                   kahan::Bool=false)

Run the Stochastic Simulation Algorithm (Direct Method) for the Telegraph model.

# Arguments
- `model`: TelegraphModel specification
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
- `counts::Vector{Int}`: Molecule counts after each event
- `states::Vector{Int}`: Promoter states after each event (0=OFF, 1=ON)
- `n_events::Int`: Total number of events
- `model::TelegraphModel`: Model specification used
- `cfg::NamedTuple`: Configuration (Tprop, Tacc, mode, kahan)
"""
function ssa_telegraph!(model::TelegraphModel, t_end::Real;
                        Tprop::Type=Float64,
                        Tacc::Type=Float32,
                        rng::AbstractRNG=Random.default_rng(),
                        mode::Symbol=:mixed,
                        kahan::Bool=false)
    if mode === :strict
        return _ssa_telegraph_strict!(model, t_end; Tprop, rng, kahan)
    else
        return _ssa_telegraph_mixed!(model, t_end; Tprop, Tacc, rng)
    end
end

function _ssa_telegraph_mixed!(model::TelegraphModel, t_end::Real;
                                Tprop::Type=Float64,
                                Tacc::Type=Float32,
                                rng::AbstractRNG=Random.default_rng())
    x = model.initial_population
    S = model.initial_state
    t = 0.0

    times = Float64[0.0]
    counts = Int[x]
    states = Int[S]

    n_events = 0
    while t < t_end
        a1, a2, a3, a4 = compute_propensities(model, S, x, Tprop)
        a0 = Tacc(a1) + Tacc(a2) + Tacc(a3) + Tacc(a4)

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
        cumsum = Float32(a1)

        reaction_idx = if cumsum >= threshold
            1
        elseif (cumsum += Float32(a2)) >= threshold
            2
        elseif (cumsum += Float32(a3)) >= threshold
            3
        else
            4
        end

        S, x = apply_reaction(model, S, x, reaction_idx)
        n_events += 1
        push!(times, t)
        push!(counts, x)
        push!(states, S)
    end

    return (times=times, counts=counts, states=states, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tacc, mode=:mixed, kahan=false))
end

function _ssa_telegraph_strict!(model::TelegraphModel, t_end::Real;
                                 Tprop::Type=Float64,
                                 rng::AbstractRNG=Random.default_rng(),
                                 kahan::Bool=false)
    x = model.initial_population
    S = model.initial_state
    t = zero(Tprop)
    t_end_T = Tprop(t_end)

    times = Float64[0.0]
    counts = Int[x]
    states = Int[S]

    kacc = kahan ? KahanAccumulator{Tprop}() : nothing

    n_events = 0
    while t < t_end_T
        a1, a2, a3, a4 = compute_propensities(model, S, x, Tprop)
        a0 = a1 + a2 + a3 + a4

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
        cs = a1
        reaction_idx = if cs >= threshold
            1
        elseif (cs += a2) >= threshold
            2
        elseif (cs += a3) >= threshold
            3
        else
            4
        end

        S, x = apply_reaction(model, S, x, reaction_idx)
        n_events += 1
        push!(times, Float64(t))
        push!(counts, x)
        push!(states, S)
    end

    return (times=times, counts=counts, states=states, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tprop, mode=:strict, kahan=kahan))
end
