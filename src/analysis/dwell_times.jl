# Dwell time analysis utilities

"""
    extract_dwell_times(states, times) -> (on_dwells, off_dwells)

Extract dwell times from a binary state trajectory. Walks the trajectory
detecting transitions where `states[i] != states[i-1]`, recording the
duration spent in the state that just ended.

State `1` dwells go into `on_dwells`, state `0` dwells go into `off_dwells`.
Includes both the first segment (from `times[1]`) and the final segment
(to `times[end]`).
"""
function extract_dwell_times(states, times)
    n = length(states)
    on_dwells  = Vector{Float64}()
    off_dwells = Vector{Float64}()

    n < 2 && return (on_dwells = on_dwells, off_dwells = off_dwells)

    t_last = Float64(times[1])
    state_last = states[1]

    for i in 2:n
        if states[i] != state_last
            dwell = Float64(times[i]) - t_last
            if state_last == 1
                push!(on_dwells, dwell)
            else
                push!(off_dwells, dwell)
            end
            t_last = Float64(times[i])
            state_last = states[i]
        end
    end

    # Final segment
    dwell = Float64(times[end]) - t_last
    if state_last == 1
        push!(on_dwells, dwell)
    else
        push!(off_dwells, dwell)
    end

    return (on_dwells = on_dwells, off_dwells = off_dwells)
end

"""
    fit_exponential(data; confidence=0.95) -> (rate, lower, upper, n)

Maximum likelihood exponential rate estimate with exact chi-squared confidence
interval. MLE: `rate = 1/mean(data)`. CI uses `2*rate*sum(data) ~ Chisq(2n)`.

Returns NaN values for empty input.
"""
function fit_exponential(data; confidence=0.95)
    n = length(data)
    n == 0 && return (rate = NaN, lower = NaN, upper = NaN, n = 0)

    S = sum(Float64, data)
    rate = n / S
    α = 1.0 - confidence
    lower = quantile(Chisq(2n), α / 2) / (2S)
    upper = quantile(Chisq(2n), 1.0 - α / 2) / (2S)

    return (rate = rate, lower = lower, upper = upper, n = n)
end

"""
    well_occupancy(counts, threshold) -> (frac_low, frac_high, n_low, n_high)

Compute occupancy fractions for bistable systems (e.g. Schlögl). Counts below
`threshold` are classified as "low well", at or above as "high well".

Returns NaN fractions for empty input.
"""
function well_occupancy(counts, threshold)
    n = length(counts)
    n == 0 && return (frac_low = NaN, frac_high = NaN, n_low = 0, n_high = 0)

    n_low = count(c -> c < threshold, counts)
    n_high = n - n_low

    return (frac_low = n_low / n, frac_high = n_high / n, n_low = n_low, n_high = n_high)
end

"""
    mean_first_passage_time(counts, times, threshold) -> (mfpt, lower, upper, n_crossings)

Estimate mean first passage time from threshold-crossing events in a trajectory.
Detects crossings where consecutive counts straddle `threshold`, then computes
the mean and t-distribution CI of inter-crossing intervals.

Returns NaN values when there are insufficient crossings for statistics.
"""
function mean_first_passage_time(counts, times, threshold)
    n = length(counts)

    # Detect crossing times
    crossing_times = Float64[]
    for i in 2:n
        crossed = (counts[i-1] < threshold && counts[i] >= threshold) ||
                  (counts[i-1] >= threshold && counts[i] < threshold)
        if crossed
            push!(crossing_times, Float64(times[i]))
        end
    end

    n_crossings = length(crossing_times)
    n_crossings == 0 && return (mfpt = NaN, lower = NaN, upper = NaN, n_crossings = 0)

    # Compute passage times between consecutive crossings
    passages = diff(crossing_times)
    length(passages) == 0 && return (mfpt = NaN, lower = NaN, upper = NaN, n_crossings = n_crossings)

    if length(passages) == 1
        return (mfpt = passages[1], lower = NaN, upper = NaN, n_crossings = n_crossings)
    end

    ci = mean_with_ci(passages)
    return (mfpt = ci.mean, lower = ci.lower, upper = ci.upper, n_crossings = n_crossings)
end
