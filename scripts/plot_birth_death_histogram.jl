#!/usr/bin/env julia

"""
Generate publication-quality birth-death histogram (Figure 1) using CairoMakie.

Shows all 6 precision modes overlaid with the analytical Poisson PMF (λ=20).
The visual overlap demonstrates that all modes produce indistinguishable results.

Usage:
  julia --project=. scripts/plot_birth_death_histogram.jl
  julia --project=. scripts/plot_birth_death_histogram.jl \
    --input-jld2=results/birth_death/ensemble_data.jld2 \
    --output-stem=figures/birth_death_histogram
"""

using JLD2
using StatsBase
using Distributions
using CairoMakie
using LaTeXStrings

include("cli_args.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/birth_death/ensemble_data.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/birth_death_histogram")

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

    counts_all = [ensemble[lbl]["counts"] for lbl in labels]

    # Shared histogram edges
    lo = minimum(minimum.(counts_all))
    hi = maximum(maximum.(counts_all))
    edges = collect((lo - 0.5):1:(hi + 0.5))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    # Analytical Poisson PMF (λ = mean of FP64 baseline)
    λ = round(sum(counts_all[1]) / length(counts_all[1]); digits=1)
    k_range = lo:hi
    poisson_pmf = [pdf(Poisson(λ), k) for k in k_range]

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    colors = Makie.wong_colors()
    # Linewidths: thickest first so all colors visible through stacking
    linewidths = [2.5, 2.0, 1.8, 1.5, 1.3, 1.0]

    # Figure size: ~468×280 pt (textwidth at 6.5in, golden ratio aspect)
    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel=L"Molecule count, $n$",
        ylabel="Probability",
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        yticks=0:0.02:0.10,
        limits=((5, 42), (0, 0.105)),
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    # Plot step histograms for each precision mode
    for (i, (lbl, counts)) in enumerate(zip(labels, counts_all))
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        stairs!(ax, centers .+ 0.5, probs;
            color=colors[i],
            linewidth=linewidths[i],
            label=lbl,
        )
    end

    # Poisson PMF as black dashed line
    lines!(ax, collect(k_range), poisson_pmf;
        color=:black,
        linewidth=1.5,
        linestyle=:dash,
        label=L"Poisson($\lambda$=%$(Int(λ)))",
    )

    # Compact legend in top-right
    Legend(fig[1, 1], ax;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, -4, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=12,
        nbanks=2,
        framevisible=false,
    )

    # Save outputs
    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
