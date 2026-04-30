#!/usr/bin/env julia

"""
Generate publication-quality Schlogl strict-mode histogram using CairoMakie.

Shows FP64 baseline + STRICT BF16+SR + STRICT BF16 RTN.
FP16 strict modes excluded (cubic propensity overflow).

Usage:
  julia --project=. scripts/plot_schlogl_histogram_strict.jl
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/schlogl/ensemble_data.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/schlogl_histogram_strict")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # Plot BF16 RTN first (behind) so it doesn't obscure the other curves
    labels = [
        "STRICT BF16 RTN",
        "FP64 baseline",
        "STRICT BF16 + SR",
    ]

    missing_labels = [lbl for lbl in labels if !haskey(ensemble, lbl)]
    isempty(missing_labels) || error("Missing labels in ensemble data: $(join(missing_labels, ", "))")

    counts_all = [ensemble[lbl]["counts"] for lbl in labels]

    # Shared histogram edges — bin width of 3 for the wide bimodal distribution
    lo = minimum(minimum.(counts_all))
    hi = maximum(maximum.(counts_all))
    edges = collect((lo - 1.5):3:(hi + 1.5))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    linewidths = [1.2, 2.5, 2.0]

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

    # Plot step histograms for each strict mode
    for (i, (lbl, counts)) in enumerate(zip(labels, counts_all))
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        stairs!(ax, centers .+ 1.5, probs;
            color=PRECISION_COLORS[lbl],
            linewidth=linewidths[i],
            label=replace(lbl, "STRICT " => ""),
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
