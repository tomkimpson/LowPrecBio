#!/usr/bin/env julia

"""
Generate publication-quality telegraph marginal distribution figure using CairoMakie.

Shows all 6 mixed-precision modes overlaid. Y-axis truncated to show tail structure;
P(n=0) annotated. No analytical overlay (no closed-form PMF); FP64 is the reference.

Usage:
  julia --project=. scripts/plot_telegraph_marginals.jl
  julia --project=. scripts/plot_telegraph_marginals.jl \
    --input-jld2=results/telegraph/ensemble_data.jld2 \
    --output-stem=figures/telegraph_marginals
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/telegraph/ensemble_data.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/telegraph_marginals")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # All 6 mixed-precision modes (order: thickest/first-plotted to thinnest)
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

    # First pass: compute histograms, find tail max and P(0)
    hist_data = []
    max_tail_prob = 0.0
    for counts in counts_all
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        push!(hist_data, probs)
        if length(probs) > 1
            max_tail_prob = max(max_tail_prob, maximum(probs[2:end]))
        end
    end
    p0_fp64 = hist_data[1][1]  # P(0) for FP64 baseline

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    linewidths = [2.5, 2.0, 1.8, 1.5, 1.3, 1.0]

    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel=L"mRNA count, $n$",
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

    for (i, (lbl, probs)) in enumerate(zip(labels, hist_data))
        stairs!(ax, centers .+ 0.5, probs;
            color=PRECISION_COLORS[lbl],
            linewidth=linewidths[i],
            label=lbl,
        )
    end

    # Truncate y-axis to show tail structure; annotate P(0)
    ylim_top = max_tail_prob * 1.8
    ylims!(ax, (0, ylim_top))
    text!(ax, 1.5, ylim_top * 0.92;
        text="P(0) = $(round(p0_fp64; digits=2))",
        fontsize=11, align=(:left, :top))

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

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
