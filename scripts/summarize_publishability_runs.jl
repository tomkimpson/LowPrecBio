#!/usr/bin/env julia

"""
Aggregate publishability-plan outputs into a single long-format CSV table.

Usage:
  julia --project=. scripts/summarize_publishability_runs.jl
  julia --project=. scripts/summarize_publishability_runs.jl --input-root results/publishability
  julia --project=. scripts/summarize_publishability_runs.jl --output-csv results/publishability/publishability_summary.csv
"""

using JLD2
using Dates

include("cli_args.jl")

function read_csv_rows(path::String)
    lines = readlines(path)
    isempty(lines) && return Dict{String, String}[]
    header = split(lines[1], ",")
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ",")
        d = Dict{String, String}()
        for i in eachindex(header)
            d[header[i]] = i <= length(vals) ? vals[i] : ""
        end
        push!(rows, d)
    end
    return rows
end

function read_metadata(path::String)
    if !isfile(path)
        return Dict{String, Any}()
    end
    try
        return JLD2.load(path, "metadata")
    catch
        return Dict{String, Any}()
    end
end

function classify_run(summary_path::String, input_root::String)
    rel = relpath(dirname(summary_path), input_root)
    parts = splitpath(rel)
    category = length(parts) >= 1 ? parts[1] : ""
    study = length(parts) >= 2 ? parts[2] : ""
    run_name = length(parts) >= 3 ? parts[end] : basename(dirname(summary_path))
    return (; category, study, run_name, rel_dir=rel)
end

function vstr(x)
    x === nothing && return ""
    return string(x)
end

function to_float_string(row::Dict{String, String}, key::String)
    val = get(row, key, "")
    isempty(val) && return ""
    x = tryparse(Float64, val)
    return x === nothing ? val : string(x)
end

function write_summary_csv(outpath::String, rows::Vector{Dict{String, String}})
    mkpath(dirname(outpath))
    cols = [
        "summary_path",
        "category",
        "study",
        "run_name",
        "model_name",
        "label",
        "seed_ssa",
        "seed_sr",
        "t_end",
        "n_replicas",
        "tag",
        "commit_hash",
        "timestamp",
        "ks_pvalue",
        "wasserstein",
        "mean",
        "variance",
        "n_negative",
        "n_total",
        "A_ks_pvalue",
        "A_wasserstein",
        "D_ks_pvalue",
        "D_wasserstein",
    ]

    open(outpath, "w") do io
        println(io, join(cols, ","))
        for row in rows
            vals = [replace(get(row, c, ""), "," => ";") for c in cols]
            println(io, join(vals, ","))
        end
    end
end

function main()
    opts = parse_cli_args(ARGS)
    input_root = arg_string(opts, "input-root", "results/publishability")
    output_csv = arg_string(opts, "output-csv", joinpath(input_root, "publishability_summary.csv"))

    summary_files = String[]
    for (root, _, files) in walkdir(input_root)
        if "summary_stats.csv" in files
            push!(summary_files, joinpath(root, "summary_stats.csv"))
        end
    end
    sort!(summary_files)

    println("Found $(length(summary_files)) summary file(s) under $input_root")

    out_rows = Dict{String, String}[]
    for sf in summary_files
        meta = read_metadata(joinpath(dirname(sf), "metadata.jld2"))
        cls = classify_run(sf, input_root)
        model_name = vstr(get(meta, "model_name", ""))

        for row in read_csv_rows(sf)
            out = Dict{String, String}()
            out["summary_path"] = sf
            out["category"] = cls.category
            out["study"] = cls.study
            out["run_name"] = cls.run_name
            out["model_name"] = model_name
            out["label"] = get(row, "label", "")
            out["seed_ssa"] = vstr(get(meta, "seed_ssa", ""))
            out["seed_sr"] = vstr(get(meta, "seed_sr", ""))
            out["t_end"] = vstr(get(meta, "t_end", ""))
            out["n_replicas"] = vstr(get(meta, "n_replicas", ""))
            out["tag"] = vstr(get(meta, "tag", ""))
            out["commit_hash"] = vstr(get(meta, "commit_hash", ""))
            out["timestamp"] = vstr(get(meta, "timestamp", ""))

            out["ks_pvalue"] = to_float_string(row, "ks_pvalue")
            out["wasserstein"] = to_float_string(row, "wasserstein")
            out["mean"] = to_float_string(row, "mean")
            out["variance"] = to_float_string(row, "variance")
            out["n_negative"] = get(row, "n_negative", "")
            out["n_total"] = get(row, "n_total", "")

            out["A_ks_pvalue"] = to_float_string(row, "A_ks_pvalue")
            out["A_wasserstein"] = to_float_string(row, "A_wasserstein")
            out["D_ks_pvalue"] = to_float_string(row, "D_ks_pvalue")
            out["D_wasserstein"] = to_float_string(row, "D_wasserstein")

            push!(out_rows, out)
        end
    end

    write_summary_csv(output_csv, out_rows)
    println("Wrote $(length(out_rows)) aggregated row(s) to $output_csv")
end

main()
