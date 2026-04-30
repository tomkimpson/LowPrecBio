#!/usr/bin/env julia

"""
Generate publication-quality repressilator drift figure using CairoMakie.

Two-panel figure: Wasserstein distance vs t_end (left), -log10(KS p-value) vs t_end (right).
Shows STRICT BF16+SR and STRICT FP16+SR from horizon sweep data.

Usage:
  julia --project=. scripts/plot_repressilator_drift.jl
  julia --project=. scripts/plot_repressilator_drift.jl \
    --summary-csv=results/publishability/summary.csv \
    --output-stem=figures/repressilator_drift
"""

using CairoMakie
using LaTeXStrings

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

function build_series(rows::Vector{Dict{String, String}}, label::String)
    pts = Tuple{Float64, Float64, Float64}[]
    for r in rows
        get(r, "label", "") == label || continue
        t = parse_float(get(r, "t_end", ""))
        w = parse_float(get(r, "wasserstein", ""))
        p = parse_float(get(r, "ks_pvalue", ""))
        push!(pts, (t, w, p))
    end
    sort!(pts, by=x -> x[1])
    xs = [p[1] for p in pts]
    ws = [p[2] for p in pts]
    ps = [p[3] for p in pts]
    return xs, ws, ps
end

function main()
    opts = parse_cli_args(ARGS)
    summary_csv = arg_string(opts, "summary-csv", "results/publishability/summary.csv")
    output_stem = arg_string(opts, "output-stem", "figures/repressilator_drift")

    isfile(summary_csv) || error("Summary CSV not found: $summary_csv")

    all_rows = read_csv_rows(summary_csv)
    rep_rows = [r for r in all_rows
                if get(r, "category", "") == "sweeps" && get(r, "study", "") == "repressilator_horizon"]
    isempty(rep_rows) && error("No repressilator horizon rows found in summary CSV.")

    labels = ["STRICT BF16 + SR", "STRICT FP16 + SR"]
    colors = Dict(
        "STRICT BF16 + SR" => "#0072B2",
        "STRICT FP16 + SR" => "#009E73",
    )
    markers = Dict(
        "STRICT BF16 + SR" => :circle,
        "STRICT FP16 + SR" => :utriangle,
    )

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    fig = Figure(size=(700, 280), figure_padding=(2, 8, 2, 2))

    # ----- Left panel: Wasserstein distance -----
    ax_w = Axis(fig[1, 1],
        xlabel=L"Simulation horizon, $t_{\mathrm{end}}$",
        ylabel="Wasserstein distance",
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
        xs, ws, _ = build_series(rep_rows, label)
        isempty(xs) && continue
        scatterlines!(ax_w, xs, ws;
            color=colors[label],
            marker=markers[label],
            markersize=8,
            linewidth=1.8,
            label=replace(label, "STRICT " => ""),
        )
    end

    Legend(fig[1, 1], ax_w;
        tellwidth=false,
        tellheight=false,
        halign=:left,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=10,
        framevisible=false,
    )

    # ----- Right panel: -log10(KS p-value) -----
    ax_p = Axis(fig[1, 2],
        xlabel=L"Simulation horizon, $t_{\mathrm{end}}$",
        ylabel=L"$-\log_{10}$(KS $p$-value)",
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
        xs, _, ps = build_series(rep_rows, label)
        isempty(xs) && continue
        neg_log_p = [-log10(max(min(p, 1.0), 1e-300)) for p in ps]
        scatterlines!(ax_p, xs, neg_log_p;
            color=colors[label],
            marker=markers[label],
            markersize=8,
            linewidth=1.8,
            label=replace(label, "STRICT " => ""),
        )
    end

    # Horizontal threshold line at p = 0.05
    hlines!(ax_p, [-log10(0.05)];
        color=:black,
        linewidth=1.0,
        linestyle=:dash,
    )
    text!(ax_p, 800, -log10(0.05) + 0.08;
        text=L"$p = 0.05$",
        fontsize=10,
        color=:black,
        align=(:right, :bottom),
    )

    Legend(fig[1, 2], ax_p;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=10,
        framevisible=false,
    )

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
