# Dimerization model
# Reactions:
#   R1: 2A → D   (association, rate kf, propensity = kf * A * (A-1) / 2)
#   R2: D  → 2A  (dissociation, rate kr, propensity = kr * D)
# Conservation law: A + 2D = M₀ (constant)

"""
    DimerModel

Dimerization model with bimolecular association and unimolecular dissociation.

# Fields
- `kf::Float64`: Forward (association) rate constant
- `kr::Float64`: Reverse (dissociation) rate constant
- `initial_A::Int`: Initial monomer count
- `initial_D::Int`: Initial dimer count

# Dynamics
Two monomers combine to form a dimer (2A → D) with rate kf·A·(A-1)/2,
and dimers dissociate back into monomers (D → 2A) with rate kr·D.
The conservation law A + 2D = M₀ holds exactly throughout the trajectory.
"""
Base.@kwdef struct DimerModel
    kf::Float64 = 1e-3
    kr::Float64 = 0.1
    initial_A::Int = 100
    initial_D::Int = 0
end

"""
    compute_propensities(model::DimerModel, A::Integer, D::Integer, ::Type{T}) where T

Compute propensities for dimerization reactions.

Returns (a1, a2) as type T:
- a1 = kf * A * (A-1) / 2  (association: 2A → D)
- a2 = kr * D               (dissociation: D → 2A)
"""
function compute_propensities(model::DimerModel, A::Integer, D::Integer, ::Type{T}) where T
    a1 = T(model.kf) * T(A) * T(max(0, A - 1)) * T(0.5)
    a2 = T(model.kr) * T(D)
    return (a1, a2)
end

"""
    apply_reaction(model::DimerModel, A::Integer, D::Integer, reaction_idx::Integer) -> (Int, Int)

Apply the selected reaction to the state (A, D).
- reaction_idx == 1: 2A → D  (A -= 2, D += 1)
- reaction_idx == 2: D → 2A  (A += 2, D -= 1)

Returns updated (A, D) tuple.
"""
function apply_reaction(model::DimerModel, A::Integer, D::Integer, reaction_idx::Integer)
    if reaction_idx == 1
        return (max(0, A - 2), D + 1)       # association
    else
        return (A + 2, max(0, D - 1))        # dissociation
    end
end

"""
    ssa_dimer!(model::DimerModel, t_end::Real;
              Tprop::Type=Float64,
              Tacc::Type=Float32,
              rng::AbstractRNG=Random.default_rng(),
              mode::Symbol=:mixed,
              kahan::Bool=false)

Run the Stochastic Simulation Algorithm (Direct Method) for the Dimer model.

# Arguments
- `model`: DimerModel specification
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
- `counts_A::Vector{Int}`: Monomer counts after each event
- `counts_D::Vector{Int}`: Dimer counts after each event
- `max_deviation::Int`: Maximum deviation from conservation law A + 2D = M₀
- `n_events::Int`: Total number of events
- `model::DimerModel`: Model specification used
- `cfg::NamedTuple`: Configuration (Tprop, Tacc, mode, kahan)
"""
function ssa_dimer!(model::DimerModel, t_end::Real;
                    Tprop::Type=Float64,
                    Tacc::Type=Float32,
                    rng::AbstractRNG=Random.default_rng(),
                    mode::Symbol=:mixed,
                    kahan::Bool=false)
    if mode === :strict
        return _ssa_dimer_strict!(model, t_end; Tprop, rng, kahan)
    else
        return _ssa_dimer_mixed!(model, t_end; Tprop, Tacc, rng)
    end
end

function _ssa_dimer_mixed!(model::DimerModel, t_end::Real;
                            Tprop::Type=Float64,
                            Tacc::Type=Float32,
                            rng::AbstractRNG=Random.default_rng())
    A = model.initial_A
    D = model.initial_D
    t = 0.0
    M0 = A + 2 * D

    times = Float64[0.0]
    counts_A = Int[A]
    counts_D = Int[D]
    max_deviation = 0

    n_events = 0
    while t < t_end
        a1, a2 = compute_propensities(model, A, D, Tprop)
        a0 = Tacc(a1) + Tacc(a2)

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
        if Float32(a1) >= threshold
            reaction_idx = 1
        else
            reaction_idx = 2
        end

        A, D = apply_reaction(model, A, D, reaction_idx)
        n_events += 1

        deviation = abs(A + 2 * D - M0)
        max_deviation = max(max_deviation, deviation)

        push!(times, t)
        push!(counts_A, A)
        push!(counts_D, D)
    end

    return (times=times, counts_A=counts_A, counts_D=counts_D,
            max_deviation=max_deviation, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tacc, mode=:mixed, kahan=false))
end

function _ssa_dimer_strict!(model::DimerModel, t_end::Real;
                             Tprop::Type=Float64,
                             rng::AbstractRNG=Random.default_rng(),
                             kahan::Bool=false)
    A = model.initial_A
    D = model.initial_D
    t = zero(Tprop)
    t_end_T = Tprop(t_end)
    M0 = A + 2 * D

    times = Float64[0.0]
    counts_A = Int[A]
    counts_D = Int[D]
    max_deviation = 0

    kacc = kahan ? KahanAccumulator{Tprop}() : nothing

    n_events = 0
    while t < t_end_T
        a1, a2 = compute_propensities(model, A, D, Tprop)
        a0 = a1 + a2

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
        if a1 >= threshold
            reaction_idx = 1
        else
            reaction_idx = 2
        end

        A, D = apply_reaction(model, A, D, reaction_idx)
        n_events += 1

        deviation = abs(A + 2 * D - M0)
        max_deviation = max(max_deviation, deviation)

        push!(times, Float64(t))
        push!(counts_A, A)
        push!(counts_D, D)
    end

    return (times=times, counts_A=counts_A, counts_D=counts_D,
            max_deviation=max_deviation, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tprop, mode=:strict, kahan=kahan))
end
