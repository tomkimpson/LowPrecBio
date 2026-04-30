#!/usr/bin/env julia

"""
Generate telegraph k_on sweep — Wasserstein distance only (single panel).

Matches the CairoMakie style used for histogram plots (latex fonts, no top/right
spines, no grid, consistent legend placement, PRECISION_COLORS).

Usage:
  julia --project=. scripts/plot_telegraph_sweep.jl
  julia --project=. scripts/plot_telegraph_sweep.jl --summary-csv=results/publishability/summary.csv
"""

using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

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

function parse_kon_from_run(run_name::String)
    m = match(r"telegraph_r3_kon_(.+)$", run_name)
    m === nothing && error("Could not parse k_on from run name: $run_name")
    token = replace(m.captures[1], "p" => ".", "m" => "-")
    return parse(Float64, token)
end

function main()
    opts = parse_cli_args(ARGS)
    summary_csv = arg_string(opts, "summary-csv", "results/publishability/summary.csv")
    output_stem = arg_string(opts, "output-stem", "figures/publishability_telegraph_phase_map")

    isfile(summary_csv) || error("Summary CSV not found: $summary_csv")

    all_rows = read_csv_rows(summary_csv)
    tele_rows = [r for r in all_rows
                 if get(r, "category", "") == "sweeps" &&
                    get(r, "study", "") == "telegraph_rates"]
    isempty(tele_rows) && error("No telegraph sweep rows found.")

    labels = [
        "STRICT BF16 RTN",
        "STRICT FP16 RTN",
        "STRICT BF16 + SR",
        "STRICT FP16 + SR",
    ]

    set_theme!(theme_latexfonts())

    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel=L"k_{\text{on}}",
        ylabel=L"Wasserstein distance, $W_1$",
        xscale=log10,
        xticks=([1e-3, 1e-2, 1e-1],
                [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    for label in labels
        pts = Tuple{Float64, Float64}[]
        for r in tele_rows
            get(r, "label", "") == label || continue
            x = parse_kon_from_run(get(r, "run_name", ""))
            w = parse(Float64, get(r, "wasserstein", ""))
            push!(pts, (x, w))
        end
        sort!(pts, by=first)
        isempty(pts) && continue
        xs = [p[1] for p in pts]
        ws = [p[2] for p in pts]
        lines!(ax, xs, ws;
            color=PRECISION_COLORS[label],
            linewidth=2.0,
            label=replace(label, "STRICT " => ""),
        )
        scatter!(ax, xs, ws;
            color=PRECISION_COLORS[label],
            markersize=6,
        )
    end

    Legend(fig[1, 1], ax;
        tellwidth=false,
        tellheight=false,
        halign=:left,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=12,
        nbanks=2,
        framevisible=false,
    )

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
