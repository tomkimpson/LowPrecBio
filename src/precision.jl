# Precision type utilities for LowPrecBio
# These map symbolic precision names to numeric types

using StochasticRounding

"""
    precision_type(sym::Symbol) -> Type

Map a symbolic precision to a numeric type used for per-step arithmetic/storage.

Supported symbols:
- `:fp64` -> Float64
- `:fp32` -> Float32
- `:fp16_sr` -> Float16sr (half precision with stochastic rounding)
- `:fp16_rtn` -> Float16 (half precision with round-to-nearest)
- `:bf16_sr` -> BFloat16sr (bfloat16 with stochastic rounding)
- `:bf16_rtn` -> BFloat16 (bfloat16 with round-to-nearest)
"""
precision_type(sym::Symbol) =
    sym === :fp64      ? Float64      :
    sym === :fp32      ? Float32      :
    sym === :fp16_sr   ? Float16sr    :
    sym === :fp16_rtn  ? Float16      :
    sym === :bf16_sr   ? BFloat16sr   :
    sym === :bf16_rtn  ? BFloat16     :
    throw(ArgumentError("unknown precision $(sym)"))

"""
    accum_type(sym::Symbol) -> Type

Accumulator precision for totals and simulation time.

Supported symbols:
- `:fp64` -> Float64
- `:fp32` -> Float32
- `:fp16` -> Float16
"""
accum_type(sym::Symbol) =
    sym === :fp32 ? Float32 :
    sym === :fp64 ? Float64 :
    sym === :fp16 ? Float16 :
    throw(ArgumentError("unknown accum precision $(sym)"))

# --- DualRNG: independent streams for SSA events and stochastic rounding ---

"""
    DualRNG

Two independent Xoshiro RNG streams: one for SSA event sampling (`ssa`),
one for stochastic rounding decisions (`sr`).

Usage: pass `dual.ssa` as the `rng` argument to SSA functions and call
`activate_sr_rng!(dual)` once before the simulation to register the SR stream.
"""
Base.@kwdef struct DualRNG
    ssa::Xoshiro = Xoshiro(42)
    sr::Xoshiro  = Xoshiro(4242)
end
DualRNG(seed_ssa::Integer, seed_sr::Integer) = DualRNG(Xoshiro(seed_ssa), Xoshiro(seed_sr))

"""
    activate_sr_rng!(dual::DualRNG)

Register `dual.sr` as the global RNG used by StochasticRounding.jl for
stochastic rounding decisions, if the package exposes `setrand!`.
"""
function activate_sr_rng!(dual::DualRNG)
    if isdefined(StochasticRounding, :setrand!)
        StochasticRounding.setrand!(dual.sr)
    end
end

# --- Kahan compensated summation ---

"""
    KahanAccumulator{T}

Kahan compensated summation accumulator for type `T`. Reduces floating-point
drift when summing many small increments in reduced precision.

Fields:
- `s::T` — running sum
- `c::T` — compensation term
"""
mutable struct KahanAccumulator{T <: AbstractFloat}
    s::T
    c::T
end
KahanAccumulator{T}() where {T} = KahanAccumulator{T}(zero(T), zero(T))

"""
    kahan_add!(acc::KahanAccumulator{T}, val::T) where T

Add `val` to the accumulator using Kahan's compensated summation algorithm.
"""
function kahan_add!(acc::KahanAccumulator{T}, val::T) where T
    y = val - acc.c
    temp = acc.s + y
    acc.c = (temp - acc.s) - y
    acc.s = temp
end
