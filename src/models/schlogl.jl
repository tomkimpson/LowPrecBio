# Schlögl model (bistable autocatalytic system)
# Reactions:
#   R1: A + 2X → 3X  (rate k1, propensity k1 * [A] * n * (n-1))
#   R2: 3X → 2X + A  (rate k2, propensity k2 * n * (n-1) * (n-2))
#   R3: B → X        (rate k3, propensity k3 * [B])
#   R4: X → B        (rate k4, propensity k4 * n)
# Features: Cubic kinetics, bistability with two stable steady states

"""
    SchloglModel

Schlögl autocatalytic system exhibiting bistability.

# Fields
- `k1::Float64`: Rate constant for A + 2X → 3X
- `k2::Float64`: Rate constant for 3X → 2X + A
- `k3::Float64`: Rate constant for B → X
- `k4::Float64`: Rate constant for X → B
- `A::Float64`: Reservoir concentration of species A (held constant)
- `B::Float64`: Reservoir concentration of species B (held constant)
- `initial_population::Int`: Initial population of X

# Bistability
The model exhibits bistability for appropriate parameter choices. The steady-state
probability distribution is bimodal with a low-population and high-population mode
separated by an unstable barrier.

# Classic bistable parameters (Vellela & Qian, 2009)
Default parameters give bistable behavior with wells around n≈80 and n≈250.
"""
Base.@kwdef struct SchloglModel
    k1::Float64 = 3.0e-7
    k2::Float64 = 1.0e-4
    k3::Float64 = 1.0e-3
    k4::Float64 = 3.5
    A::Float64 = 1.0e5
    B::Float64 = 2.0e5
    initial_population::Int = 250
end

"""
    compute_propensities(model::SchloglModel, n::Integer, ::Type{T}) where T

Compute propensities for all four Schlögl reactions.

Returns (a1, a2, a3, a4) as type T:
- a1 = k1 * A * n * (n-1)     (A + 2X → 3X)
- a2 = k2 * n * (n-1) * (n-2) (3X → 2X + A)
- a3 = k3 * B                  (B → X)
- a4 = k4 * n                  (X → B)
"""
function compute_propensities(model::SchloglModel, n::Integer, ::Type{T}) where T
    n_T = T(n)

    # Propensity for R1: A + 2X → 3X
    # Rate = k1 * [A] * n * (n-1) / 2! but typically written without factorial
    a1 = T(model.k1) * T(model.A) * n_T * T(max(0, n - 1))

    # Propensity for R2: 3X → 2X + A
    # Rate = k2 * n * (n-1) * (n-2) / 3! but typically written without factorial
    a2 = T(model.k2) * n_T * T(max(0, n - 1)) * T(max(0, n - 2))

    # Propensity for R3: B → X
    a3 = T(model.k3) * T(model.B)

    # Propensity for R4: X → B
    a4 = T(model.k4) * n_T

    return (a1, a2, a3, a4)
end

"""
    apply_reaction(model::SchloglModel, n::Integer, reaction_idx::Integer) -> Int

Apply the selected reaction to the population of X.
- reaction_idx == 1: A + 2X → 3X (n → n+1)
- reaction_idx == 2: 3X → 2X + A (n → n-1)
- reaction_idx == 3: B → X (n → n+1)
- reaction_idx == 4: X → B (n → n-1)
"""
function apply_reaction(model::SchloglModel, n::Integer, reaction_idx::Integer)
    if reaction_idx == 1
        return n + 1  # A + 2X → 3X
    elseif reaction_idx == 2
        return max(0, n - 1)  # 3X → 2X + A
    elseif reaction_idx == 3
        return n + 1  # B → X
    else
        return max(0, n - 1)  # X → B
    end
end

"""
    ssa_schlogl!(model::SchloglModel, t_end::Real;
                 Tprop::Type=Float64,
                 Tacc::Type=Float32,
                 rng::AbstractRNG=Random.default_rng(),
                 mode::Symbol=:mixed,
                 kahan::Bool=false)

Run the Stochastic Simulation Algorithm (Direct Method) for the Schlögl model.

# Arguments
- `model`: SchloglModel specification
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
- `model::SchloglModel`: Model specification used
- `cfg::NamedTuple`: Configuration (Tprop, Tacc, mode, kahan)

# Notes
Strict mode with Float16 will overflow for the Schlögl model due to cubic
propensities (e.g. 250*249*248 > 65504). Use Float32 or wider for strict mode.
"""
function ssa_schlogl!(model::SchloglModel, t_end::Real;
                      Tprop::Type=Float64,
                      Tacc::Type=Float32,
                      rng::AbstractRNG=Random.default_rng(),
                      mode::Symbol=:mixed,
                      kahan::Bool=false,
                      max_events::Int=typemax(Int))
    if mode === :strict
        return _ssa_schlogl_strict!(model, t_end; Tprop, rng, kahan, max_events)
    else
        return _ssa_schlogl_mixed!(model, t_end; Tprop, Tacc, rng)
    end
end

function _ssa_schlogl_mixed!(model::SchloglModel, t_end::Real;
                              Tprop::Type=Float64,
                              Tacc::Type=Float32,
                              rng::AbstractRNG=Random.default_rng())
    n = model.initial_population
    t = 0.0

    times = Float64[0.0]
    counts = Int[n]

    n_events = 0
    while t < t_end
        a1, a2, a3, a4 = compute_propensities(model, n, Tprop)
        a0 = Tacc(a1) + Tacc(a2) + Tacc(a3) + Tacc(a4)

        if a0 <= zero(Tacc) || !isfinite(a0)
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

        n = apply_reaction(model, n, reaction_idx)
        n_events += 1
        push!(times, t)
        push!(counts, n)
    end

    return (times=times, counts=counts, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tacc, mode=:mixed, kahan=false))
end

function _ssa_schlogl_strict!(model::SchloglModel, t_end::Real;
                               Tprop::Type=Float64,
                               rng::AbstractRNG=Random.default_rng(),
                               kahan::Bool=false,
                               max_events::Int=typemax(Int))
    n = model.initial_population
    t = zero(Tprop)
    t_end_T = Tprop(t_end)

    times = Float64[0.0]
    counts = Int[n]

    kacc = kahan ? KahanAccumulator{Tprop}() : nothing

    n_events = 0
    while t < t_end_T && n_events < max_events
        a1, a2, a3, a4 = compute_propensities(model, n, Tprop)
        a0 = a1 + a2 + a3 + a4

        if a0 <= zero(Tprop) || !isfinite(a0)
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

        n = apply_reaction(model, n, reaction_idx)
        n_events += 1
        push!(times, Float64(t))
        push!(counts, n)
    end

    return (times=times, counts=counts, n_events=n_events, model=model,
            cfg=(Tprop=Tprop, Tacc=Tprop, mode=:strict, kahan=kahan))
end
