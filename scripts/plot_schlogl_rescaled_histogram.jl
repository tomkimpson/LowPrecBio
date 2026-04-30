#!/usr/bin/env julia

"""
Generate publication-quality histogram for the rescaled Schlogl experiment.

Shows all 6 mixed-precision modes (including FP16) overlaid, demonstrating
that reparameterization brings FP16 into representable range.

Usage:
  julia --project=. scripts/plot_schlogl_rescaled_histogram.jl
  julia --project=. scripts/plot_schlogl_rescaled_histogram.jl \
    --input-jld2=results/schlogl_rescaled/ensemble_data.jld2 \
    --output-stem=figures/schlogl_rescaled_histogram
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/schlogl_rescaled/ensemble_data.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/schlogl_rescaled_histogram")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # All 6 mixed-precision modes (FP16 now included thanks to rescaling)
    labels = [
        "FP64 baseline",
        "FP32",
        "BF16 + SR",
        "FP16 + SR",
        "BF16 RTN",
        "FP16 RTN",
    ]

    # Filter to labels actually present in the data (mixed modes only; strict in separate panel)
    available_labels = [lbl for lbl in labels if haskey(ensemble, lbl)]

    isempty(available_labels) && error("No matching labels found in ensemble data")

    counts_all = [ensemble[lbl]["counts"] for lbl in available_labels]

    # Shared histogram edges — bin width of 3 for the wide bimodal distribution
    lo = minimum(minimum.(counts_all))
    hi = maximum(maximum.(counts_all))
    edges = collect((lo - 1.5):3:(hi + 1.5))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    n = length(available_labels)
    linewidths = range(2.5, 1.0; length=n)

    # Figure size: ~468x280 pt (textwidth at 6.5in, golden ratio aspect)
    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel=L"Molecule count, $n$",
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

    # Plot step histograms for each precision mode
    for (i, (lbl, counts)) in enumerate(zip(available_labels, counts_all))
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        stairs!(ax, centers .+ 1.5, probs;
            color=PRECISION_COLORS[lbl],
            linewidth=linewidths[i],
            label=lbl,
        )
    end

    # Compact legend in top-right
    Legend(fig[1, 1], ax;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
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
