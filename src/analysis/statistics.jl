# Statistical analysis utilities

"""
    wasserstein_distance(s1, s2) -> Float64

Compute the 1-Wasserstein (earth mover's) distance between two samples.
Sorts combined samples and integrates |F1 - F2| piecewise over intervals
where both ECDFs are constant.
"""
function wasserstein_distance(s1, s2)
    x = sort(collect(Float64, s1))
    y = sort(collect(Float64, s2))
    n1, n2 = length(x), length(y)
    grid = sort(unique(vcat(x, y)))
    length(grid) <= 1 && return 0.0
    dist = 0.0
    for k in 1:(length(grid) - 1)
        # F1(t) for t in [grid[k], grid[k+1]) = fraction of x ≤ grid[k]
        f1 = searchsortedlast(x, grid[k]) / n1
        f2 = searchsortedlast(y, grid[k]) / n2
        dx = grid[k+1] - grid[k]
        dist += abs(f1 - f2) * dx
    end
    return dist
end

"""
    ks_two_sample_test(s1, s2) -> (statistic, p_value)

Two-sample Kolmogorov-Smirnov test via HypothesisTests.
"""
function ks_two_sample_test(s1, s2)
    t = HypothesisTests.ApproximateTwoSampleKSTest(collect(Float64, s1), collect(Float64, s2))
    return (statistic = t.δ, p_value = pvalue(t))
end

"""
    mean_with_ci(data; confidence=0.95) -> (mean, lower, upper)

Sample mean with t-distribution confidence interval.
"""
function mean_with_ci(data; confidence=0.95)
    n = length(data)
    m = mean(data)
    s = std(data)
    α = 1.0 - confidence
    t_crit = quantile(TDist(n - 1), 1.0 - α / 2)
    margin = t_crit * s / sqrt(n)
    return (mean = m, lower = m - margin, upper = m + margin)
end

"""
    variance_with_ci(data; confidence=0.95) -> (variance, lower, upper)

Sample variance with chi-squared confidence interval.
"""
function variance_with_ci(data; confidence=0.95)
    n = length(data)
    v = var(data)
    α = 1.0 - confidence
    χ2_lower = quantile(Chisq(n - 1), α / 2)
    χ2_upper = quantile(Chisq(n - 1), 1.0 - α / 2)
    return (variance = v, lower = (n - 1) * v / χ2_upper, upper = (n - 1) * v / χ2_lower)
end

"""
    negative_population_count(counts) -> Int

Count the number of negative values in a population counts array.
"""
function negative_population_count(counts)
    return count(x -> x < 0, counts)
end

"""
    underflow_overflow_counts(values, T) -> (n_underflow, n_overflow)

Count values that would underflow or overflow when represented in type `T`.
"""
function underflow_overflow_counts(values, T)
    fmin = Float64(floatmin(T))
    fmax = Float64(floatmax(T))
    n_under = count(x -> 0 < abs(Float64(x)) < fmin, values)
    n_over  = count(x -> abs(Float64(x)) > fmax, values)
    return (n_underflow = n_under, n_overflow = n_over)
end

"""
    anderson_darling_test(data, dist) -> (statistic, p_value)

Anderson-Darling goodness-of-fit test via HypothesisTests.
"""
function anderson_darling_test(data, dist)
    t = HypothesisTests.OneSampleADTest(collect(Float64, data), dist)
    return (statistic = t.A², p_value = pvalue(t))
end

"""
    reaction_channel_frequencies(events, n_channels) -> Vector{Float64}

Compute normalized frequency of each reaction channel from event labels.
"""
function reaction_channel_frequencies(events, n_channels)
    counts = zeros(Int, n_channels)
    for e in events
        counts[e] += 1
    end
    total = sum(counts)
    total == 0 && return zeros(Float64, n_channels)
    return counts ./ total
end
