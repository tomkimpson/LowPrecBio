#!/usr/bin/env julia

"""
Generate publication-facing sweep figures from publishability summary CSV.

Usage:
  julia --project=. scripts/plot_publishability_sweeps.jl
  julia --project=. scripts/plot_publishability_sweeps.jl --summary-csv=results/publishability/summary.csv
  julia --project=. scripts/plot_publishability_sweeps.jl --output-dir=figures
"""

using Printf
using Plots

include("cli_args.jl")

function read_csv_rows(path::String)
    lines = readlines(path)
    isempty(lines) && return Dict{String, String}[]
    header = split(lines[1], ",")
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ","; keepempty=true)
        d = Dict{String, String}()
        for i in eachindex(header)
            d[header[i]] = i <= length(vals) ? vals[i] : ""
        end
        push!(rows, d)
    end
    return rows
end

function parse_float(s::String)
    x = tryparse(Float64, s)
    x === nothing && error("Expected float, got: '$s'")
    return x
end

function parse_kon_from_run(run_name::String)
    m = match(r"telegraph_r3_kon_(.+)$", run_name)
    m === nothing && error("Could not parse k_on from run name: $run_name")
    token = m.captures[1]
    token = replace(token, "p" => ".")
    token = replace(token, "m" => "-")
    return parse_float(token)
end

function filter_rows(rows::Vector{Dict{String, String}}; study::String)
    return [r for r in rows if get(r, "category", "") == "sweeps" && get(r, "study", "") == study]
end

function build_series(rows::Vector{Dict{String, String}}, label::String;
                      x_key::String, x_from_run::Bool=false)
    pts = Tuple{Float64, Float64, Float64}[]
    for r in rows
        get(r, "label", "") == label || continue
        x = x_from_run ? parse_kon_from_run(get(r, "run_name", "")) : parse_float(get(r, x_key, ""))
        w = parse_float(get(r, "wasserstein", ""))
        p = parse_float(get(r, "ks_pvalue", ""))
        push!(pts, (x, w, p))
    end
    sort!(pts, by=x -> x[1])
    xs = [p[1] for p in pts]
    ws = [p[2] for p in pts]
    ps = [p[3] for p in pts]
    return xs, ws, ps
end

function safe_log10_pvals(ps::Vector{Float64})
    return [-log10(max(min(p, 1.0), 1e-300)) for p in ps]
end

function plot_telegraph(rows::Vector{Dict{String, String}})
    tele_rows = filter_rows(rows; study="telegraph_rates")
    isempty(tele_rows) && error("No telegraph sweep rows found in summary CSV.")

    labels = [
        "STRICT BF16 RTN",
        "STRICT FP16 RTN",
        "STRICT BF16 + SR",
        "STRICT FP16 + SR",
    ]
    colors = Dict(
        "STRICT BF16 RTN" => "#D55E00",
        "STRICT FP16 RTN" => "#E69F00",
        "STRICT BF16 + SR" => "#0072B2",
        "STRICT FP16 + SR" => "#009E73",
    )

    p_w = plot(
        title="Telegraph Rare-Event Stress (Strict Modes)",
        xlabel="k_on (OFF->ON rate)",
        ylabel="Wasserstein vs FP64",
        xscale=:log10,
        legend=:topleft,
        lw=2,
    )
    p_p = plot(
        title="Telegraph KS Significance (Strict Modes)",
        xlabel="k_on (OFF->ON rate)",
        ylabel="-log10(KS p-value)",
        xscale=:log10,
        legend=:topleft,
        lw=2,
    )

    for label in labels
        xs, ws, ps = build_series(tele_rows, label; x_key="k_on", x_from_run=true)
        isempty(xs) && continue
        plot!(p_w, xs, ws; marker=:circle, label=replace(label, "STRICT " => ""), color=colors[label])
        plot!(p_p, xs, safe_log10_pvals(ps); marker=:circle, label=replace(label, "STRICT " => ""), color=colors[label])
    end

    hline!(p_p, [-(log10(0.05))]; linestyle=:dash, color=:black, label="p=0.05")
    return plot(p_w, p_p; layout=(1, 2), size=(1300, 460))
end

function plot_repressilator(rows::Vector{Dict{String, String}})
    rep_rows = filter_rows(rows; study="repressilator_horizon")
    isempty(rep_rows) && error("No repressilator horizon rows found in summary CSV.")

    labels = ["STRICT BF16 + SR", "STRICT FP16 + SR"]
    colors = Dict(
        "STRICT BF16 + SR" => "#0072B2",
        "STRICT FP16 + SR" => "#009E73",
    )

    p_w = plot(
        title="Repressilator Horizon Drift (Strict Modes)",
        xlabel="t_end",
        ylabel="Wasserstein vs FP64",
        legend=:topleft,
        lw=2,
    )
    p_p = plot(
        title="Repressilator KS Significance (Strict Modes)",
        xlabel="t_end",
        ylabel="-log10(KS p-value)",
        legend=:topleft,
        lw=2,
    )

    for label in labels
        xs, ws, ps = build_series(rep_rows, label; x_key="t_end", x_from_run=false)
        isempty(xs) && continue
        plot!(p_w, xs, ws; marker=:circle, label=replace(label, "STRICT " => ""), color=colors[label])
        plot!(p_p, xs, safe_log10_pvals(ps); marker=:circle, label=replace(label, "STRICT " => ""), color=colors[label])
    end

    hline!(p_p, [-(log10(0.05))]; linestyle=:dash, color=:black, label="p=0.05")
    return plot(p_w, p_p; layout=(1, 2), size=(1300, 460))
end

function save_dual_formats(fig, out_stem::String)
    savefig(fig, out_stem * ".png")
    savefig(fig, out_stem * ".pdf")
end

function main()
    opts = parse_cli_args(ARGS)
    summary_csv = arg_string(opts, "summary-csv", "results/publishability/summary.csv")
    output_dir = arg_string(opts, "output-dir", "figures")

    isfile(summary_csv) || error("Summary CSV not found: $summary_csv")
    mkpath(output_dir)

    rows = read_csv_rows(summary_csv)
    isempty(rows) && error("Summary CSV has no rows: $summary_csv")

    tele_fig = plot_telegraph(rows)
    rep_fig = plot_repressilator(rows)

    tele_stem = joinpath(output_dir, "publishability_telegraph_phase_map")
    rep_stem = joinpath(output_dir, "publishability_repressilator_drift")

    save_dual_formats(tele_fig, tele_stem)
    save_dual_formats(rep_fig, rep_stem)

    println("Saved:")
    println("  ", tele_stem, ".png")
    println("  ", tele_stem, ".pdf")
    println("  ", rep_stem, ".png")
    println("  ", rep_stem, ".pdf")
end

main()
