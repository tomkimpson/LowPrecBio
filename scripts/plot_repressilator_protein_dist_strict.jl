#!/usr/bin/env julia

"""
Generate repressilator protein distribution (strict precision panel) using CairoMakie.

Plots RTN modes behind (thinner) so they don't obscure the other curves.

Usage:
  julia --project=. scripts/plot_repressilator_protein_dist_strict.jl
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/repressilator/ensemble_data.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/repressilator_protein_dist_strict")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    ensemble = data["ensemble"]

    # Plot RTN modes first (behind) so they don't obscure the other curves
    plot_order = [
        "STRICT BF16 RTN",
        "STRICT FP16 RTN",
        "FP64 baseline",
        "STRICT BF16 + SR",
        "STRICT FP16 + SR",
    ]
    # Legend order: FP64 first
    labels = [
        "FP64 baseline",
        "STRICT BF16 + SR",
        "STRICT FP16 + SR",
        "STRICT BF16 RTN",
        "STRICT FP16 RTN",
    ]

    all_labels = union(plot_order, labels)
    counts_all = Dict(lbl => ensemble[lbl]["end_pA"] for lbl in all_labels)

    all_counts = collect(values(counts_all))
    lo = minimum(minimum.(all_counts))
    hi = maximum(maximum.(all_counts))
    edges = collect((lo - 2.5):5:(hi + 2.5))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    set_theme!(theme_latexfonts())

    plot_linewidths = Dict(
        "STRICT BF16 RTN" => 1.2, "STRICT FP16 RTN" => 1.2,
        "FP64 baseline" => 2.5, "STRICT BF16 + SR" => 2.0, "STRICT FP16 + SR" => 1.8,
    )

    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel=L"Protein A count, $n_{pA}$",
        ylabel="Probability",
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    # Plot in draw order (RTN behind)
    plot_elements = Dict{String, Any}()
    for lbl in plot_order
        h = fit(Histogram, counts_all[lbl], edges)
        probs = h.weights ./ sum(h.weights)
        p = stairs!(ax, centers .+ 2.5, probs;
            color=PRECISION_COLORS[lbl],
            linewidth=plot_linewidths[lbl],
        )
        plot_elements[lbl] = p
    end

    # Build legend in desired order (FP64 first)
    legend_entries = [plot_elements[lbl] for lbl in labels]
    legend_labels = [replace(lbl, "STRICT " => "") for lbl in labels]

    Legend(fig[1, 1], legend_entries, legend_labels;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=12,
        nbanks=1,
        framevisible=false,
    )

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
