# I/O and data serialization utilities

"""
    model_to_dict(model) -> Dict{String, Any}

Convert any `@kwdef` model struct to a plain `Dict{String,Any}` using
`fieldnames`. Values that are types are stored as their string representation
so that loading does not require the original type to be in scope.
"""
function model_to_dict(model)
    d = Dict{String, Any}()
    for f in fieldnames(typeof(model))
        v = getfield(model, f)
        d[string(f)] = v
    end
    return d
end

"""
    compute_validation_metrics(results, reference_key; count_field=:counts)

Compute validation metrics (Wasserstein distance, KS test, mean/variance CIs,
negative count fraction) for all keys in `results` vs the `reference_key`.

`count_field` can be a single `Symbol` or a `Vector{Symbol}` for models with
multiple count fields (e.g. Dimer has `:A` and `:D`).

Returns `Dict{String, Dict{String, Any}}` keyed by label.
"""
function compute_validation_metrics(results::Dict, reference_key::String;
                                    count_field=:counts)
    fields = count_field isa Symbol ? [count_field] : collect(count_field)
    ref = results[reference_key]
    metrics = Dict{String, Dict{String, Any}}()

    for (label, res) in results
        m = Dict{String, Any}("label" => label)

        for cf in fields
            prefix = length(fields) > 1 ? "$(cf)_" : ""
            ref_data = _get_count_data(ref, cf)
            test_data = _get_count_data(res, cf)

            # Wasserstein distance
            m["$(prefix)wasserstein"] = wasserstein_distance(ref_data, test_data)

            # KS test
            ks = ks_two_sample_test(ref_data, test_data)
            m["$(prefix)ks_statistic"] = ks.statistic
            m["$(prefix)ks_pvalue"] = ks.p_value

            # Mean with CI
            mci = mean_with_ci(test_data)
            m["$(prefix)mean"] = mci.mean
            m["$(prefix)mean_lower"] = mci.lower
            m["$(prefix)mean_upper"] = mci.upper

            # Variance with CI
            vci = variance_with_ci(test_data)
            m["$(prefix)variance"] = vci.variance
            m["$(prefix)var_lower"] = vci.lower
            m["$(prefix)var_upper"] = vci.upper

            # Negative counts
            m["$(prefix)n_negative"] = negative_population_count(test_data)
            m["$(prefix)n_total"] = length(test_data)
        end

        metrics[label] = m
    end

    return metrics
end

"""
    _get_count_data(result, field::Symbol) -> Vector

Extract count data from a result NamedTuple or Dict by field name.
"""
function _get_count_data(result, field::Symbol)
    if result isa Dict
        return result[string(field)]
    else
        return getfield(result, field)
    end
end

"""
    save_ensemble_data(filepath, results)

Save ensemble results to a JLD2 file. NamedTuples are converted to
`Dict{String,Any}` to avoid JLD2 anonymous-type deserialization issues.
"""
function save_ensemble_data(filepath, results::Dict)
    data = Dict{String, Any}()
    for (label, res) in results
        data[label] = _namedtuple_to_dict(res)
    end
    JLD2.save(filepath, "ensemble", data)
end

"""
    _namedtuple_to_dict(nt) -> Dict{String, Any}

Convert a NamedTuple (or Dict) to Dict{String, Any} for serialization.
"""
function _namedtuple_to_dict(nt)
    if nt isa Dict
        return nt
    end
    d = Dict{String, Any}()
    for k in keys(nt)
        d[string(k)] = nt[k]
    end
    return d
end

"""
    save_metadata(filepath; model_name, model_params, t_end, n_replicas,
                  seed_ssa, seed_sr, configs, timestamp=now(),
                  extra_metadata=nothing)

Save experiment metadata (reproducibility info) to a JLD2 file.
If `extra_metadata` is provided, keys are merged into the metadata dictionary.
"""
function save_metadata(filepath;
                       model_name::String,
                       model_params::Dict{String, Any},
                       t_end,
                       n_replicas::Int,
                       seed_ssa::Int,
                       seed_sr::Int,
                       configs,
                       timestamp=Dates.now(),
                       extra_metadata=nothing)
    meta = Dict{String, Any}(
        "model_name"   => model_name,
        "model_params" => model_params,
        "t_end"        => Float64(t_end),
        "n_replicas"   => n_replicas,
        "seed_ssa"     => seed_ssa,
        "seed_sr"      => seed_sr,
        "configs"      => _serialize_configs(configs),
        "timestamp"    => string(timestamp),
    )
    if extra_metadata !== nothing
        for (k, v) in extra_metadata
            meta[string(k)] = v
        end
    end
    JLD2.save(filepath, "metadata", meta)
end

"""
    _serialize_configs(configs) -> Vector{Dict{String,Any}}

Convert config tuples to plain dicts for serialization.
Handles both 3-tuple (label, prec, acc) and 4-tuple (label, prec, acc, mode) forms.
"""
function _serialize_configs(configs)
    out = Vector{Dict{String, Any}}()
    for c in configs
        d = Dict{String, Any}("label" => string(c[1]))
        if length(c) >= 2
            d["precision"] = string(c[2])
        end
        if length(c) >= 3
            d["accumulator"] = string(c[3])
        end
        if length(c) >= 4
            d["mode"] = string(c[4])
        end
        push!(out, d)
    end
    return out
end

"""
    save_summary_csv(filepath, metrics; labels_order=nothing)

Write a human-readable CSV summary table from validation metrics.
If `labels_order` is provided, rows are written in that order;
otherwise alphabetical.
"""
function save_summary_csv(filepath, metrics::Dict;
                          labels_order::Union{Nothing, Vector{String}}=nothing)
    labels = labels_order === nothing ? sort(collect(keys(metrics))) : labels_order

    # Determine columns from first entry
    first_m = metrics[first(labels)]
    # Exclude "label" from columns
    cols = sort(filter(k -> k != "label", collect(keys(first_m))))

    open(filepath, "w") do io
        # Header
        print(io, "label")
        for c in cols
            print(io, ",", c)
        end
        println(io)

        # Data rows
        for label in labels
            m = metrics[label]
            print(io, label)
            for c in cols
                v = get(m, c, "")
                if v isa AbstractFloat
                    @printf(io, ",%.8g", v)
                elseif v isa Integer
                    print(io, ",", v)
                else
                    print(io, ",", v)
                end
            end
            println(io)
        end
    end
end

"""
    save_results(model_name, results; model, t_end, n_replicas,
                 seed_ssa, seed_sr, configs,
                 reference_key="FP64 baseline", count_field=:counts,
                 extra_metrics=nothing, output_dir="results",
                 extra_metadata=nothing)

Convenience wrapper that saves ensemble data, metadata, validation metrics,
and a summary CSV to `output_dir/model_name/`.

`extra_metrics` is an optional `Dict{String, Any}` of additional data to save
alongside the standard files (saved as `extra_metrics.jld2`).
"""
function save_results(model_name::String, results::Dict;
                      model,
                      t_end,
                      n_replicas::Int,
                      seed_ssa::Int,
                      seed_sr::Int,
                      configs,
                      reference_key::String="FP64 baseline",
                      count_field=:counts,
                      extra_metrics=nothing,
                      output_dir::String="results",
                      extra_metadata=nothing)
    dir = joinpath(output_dir, model_name)
    mkpath(dir)

    # 1. Save raw ensemble data
    save_ensemble_data(joinpath(dir, "ensemble_data.jld2"), results)

    # 2. Save metadata
    save_metadata(joinpath(dir, "metadata.jld2");
                  model_name=model_name,
                  model_params=model_to_dict(model),
                  t_end=t_end,
                  n_replicas=n_replicas,
                  seed_ssa=seed_ssa,
                  seed_sr=seed_sr,
                  configs=configs,
                  extra_metadata=extra_metadata)

    # 3. Compute and save validation metrics
    metrics = compute_validation_metrics(results, reference_key; count_field=count_field)

    # Build label ordering from configs
    labels_order = String[]
    for c in configs
        label = string(c[1])
        if haskey(metrics, label)
            push!(labels_order, label)
        end
    end
    # Add any remaining labels not in configs (e.g. strict modes added separately)
    for label in sort(collect(keys(metrics)))
        if !(label in labels_order)
            push!(labels_order, label)
        end
    end

    save_summary_csv(joinpath(dir, "summary_stats.csv"), metrics; labels_order=labels_order)

    # 4. Save extra metrics if provided
    if extra_metrics !== nothing
        JLD2.save(joinpath(dir, "extra_metrics.jld2"), "extra", extra_metrics)
    end

    println("Results saved to $(dir)/")
    println("  - ensemble_data.jld2  ($(length(results)) precision modes)")
    println("  - metadata.jld2       (seeds, params, timestamp)")
    println("  - summary_stats.csv   ($(length(metrics)) rows)")
    if extra_metrics !== nothing
        println("  - extra_metrics.jld2  ($(length(extra_metrics)) entries)")
    end

    return dir
end

"""
    load_ensemble_data(filepath) -> Dict

Load ensemble data from a JLD2 file. Returns the dictionary of results
keyed by precision label.
"""
function load_ensemble_data(filepath)
    return JLD2.load(filepath, "ensemble")
end
