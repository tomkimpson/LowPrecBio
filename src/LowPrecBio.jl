module LowPrecBio

using Random, Statistics, StatsBase
using Distributions
using HypothesisTests
using StochasticRounding
using BenchmarkTools
using JLD2, Dates, Printf

# Precision utilities
include("precision.jl")
export precision_type, accum_type
export DualRNG, activate_sr_rng!
export KahanAccumulator, kahan_add!

# Models
include("models/birth_death.jl")
export BirthDeathModel, ssa_birth_death!

include("models/schlogl.jl")
export SchloglModel, ssa_schlogl!

include("models/telegraph.jl")
export TelegraphModel, ssa_telegraph!

include("models/dimer.jl")
export DimerModel, ssa_dimer!

include("models/repressilator.jl")
export RepressilatorModel, ssa_repressilator!

# Analysis utilities
include("analysis/statistics.jl")
export wasserstein_distance, ks_two_sample_test
export mean_with_ci, variance_with_ci
export negative_population_count, underflow_overflow_counts
export anderson_darling_test, reaction_channel_frequencies

include("analysis/dwell_times.jl")
export extract_dwell_times, fit_exponential
export well_occupancy, mean_first_passage_time

# I/O and serialization
include("io.jl")
export save_results, load_ensemble_data
export compute_validation_metrics, model_to_dict

# Benchmarking utilities
include("benchmarks.jl")
export BenchmarkResult
export benchmark_single, benchmark_ensemble, benchmark_model
export save_benchmark_results, save_benchmark_summary

end
