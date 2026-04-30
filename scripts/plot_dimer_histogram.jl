#!/usr/bin/env julia

"""
Generate publication-quality dimer histogram figure using CairoMakie.

Two-panel figure: monomer A (left), dimer D (right).
Shows all 6 precision modes overlaid as stairs histograms.

Usage:
  julia --project=. scripts/plot_dimer_histogram.jl
  julia --project=. scripts/plot_dimer_histogram.jl \
    --input-jld2=results/dimer/ensemble_data.jld2 \
    --output-stem=figures/dimer_histogram
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/dimer/ensemble_data.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/dimer_histogram")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # All 6 precision modes (order: thickest/first-plotted to thinnest)
    labels = [
        "FP64 baseline",
        "FP32",
        "BF16 + SR",
        "FP16 + SR",
        "BF16 RTN",
        "FP16 RTN",
    ]

    missing_labels = [lbl for lbl in labels if !haskey(ensemble, lbl)]
    isempty(missing_labels) || error("Missing labels in ensemble data: $(join(missing_labels, ", "))")

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    colors = Makie.wong_colors()
    linewidths = [2.5, 2.0, 1.8, 1.5, 1.3, 1.0]

    fig = Figure(size=(700, 280), figure_padding=(2, 8, 2, 2))

    # ----- Left panel: Monomer A -----
    counts_A = [ensemble[lbl]["A"] for lbl in labels]
    lo_A = minimum(minimum.(counts_A))
    hi_A = maximum(maximum.(counts_A))
    edges_A = collect((lo_A - 1.0):2:(hi_A + 1.0))
    centers_A = (edges_A[1:end-1] .+ edges_A[2:end]) ./ 2

    ax_A = Axis(fig[1, 1],
        xlabel=L"Monomer count, $n_A$",
        ylabel="Probability",
        title="Monomer (A)",
        titlesize=12,
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    for (i, (lbl, counts)) in enumerate(zip(labels, counts_A))
        h = fit(Histogram, counts, edges_A)
        probs = h.weights ./ sum(h.weights)
        stairs!(ax_A, centers_A .+ 1.0, probs;
            color=colors[i],
            linewidth=linewidths[i],
            label=lbl,
        )
    end

    Legend(fig[1, 1], ax_A;
        tellwidth=false,
        tellheight=false,
        halign=:left,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=10,
        nbanks=1,
        framevisible=false,
    )

    # ----- Right panel: Dimer D -----
    counts_D = [ensemble[lbl]["D"] for lbl in labels]
    lo_D = minimum(minimum.(counts_D))
    hi_D = maximum(maximum.(counts_D))
    edges_D = collect((lo_D - 0.5):1:(hi_D + 0.5))
    centers_D = (edges_D[1:end-1] .+ edges_D[2:end]) ./ 2

    ax_D = Axis(fig[1, 2],
        xlabel=L"Dimer count, $n_D$",
        ylabel="",
        title="Dimer (D)",
        titlesize=12,
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    for (i, (lbl, counts)) in enumerate(zip(labels, counts_D))
        h = fit(Histogram, counts, edges_D)
        probs = h.weights ./ sum(h.weights)
        stairs!(ax_D, centers_D .+ 0.5, probs;
            color=colors[i],
            linewidth=linewidths[i],
            label=lbl,
        )
    end

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
