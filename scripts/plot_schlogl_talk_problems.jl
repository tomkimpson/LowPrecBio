#!/usr/bin/env julia

"""
Generate combined Schlogl histogram for talk — problem cases.

Shows FP64 baseline + FP32 + BF16 RTN (mixed) + STRICT BF16 RTN (faint overlay).
The strict curve is drawn with low opacity to show the contrast without obscuring.

Usage:
  julia --project=. scripts/plot_schlogl_talk_problems.jl
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
    output_stem = arg_string(opts, "output-stem", "figures/schlogl_talk_problems")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # Main curves (full opacity)
    main_labels = [
        "FP64 baseline",
        "FP32",
        "BF16 RTN",
    ]
    # Faint overlay
    overlay_label = "STRICT BF16 RTN"

    all_labels = vcat(main_labels, [overlay_label])
    missing_labels = [lbl for lbl in all_labels if !haskey(ensemble, lbl)]
    isempty(missing_labels) || error("Missing labels in ensemble data: $(join(missing_labels, ", "))")

    main_counts = [ensemble[lbl]["counts"] for lbl in main_labels]
    overlay_counts = ensemble[overlay_label]["counts"]

    # Shared histogram edges
    all_counts = vcat(main_counts, [overlay_counts])
    lo = minimum(minimum.(all_counts))
    hi = maximum(maximum.(all_counts))
    edges = collect((lo - 1.5):3:(hi + 1.5))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    set_theme!(theme_latexfonts())

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

    # Plot FP64 first so it takes top-left position in the 2-column legend
    h_fp64 = fit(Histogram, main_counts[1], edges)
    probs_fp64 = h_fp64.weights ./ sum(h_fp64.weights)
    stairs!(ax, centers .+ 1.5, probs_fp64;
        color=PRECISION_COLORS[main_labels[1]],
        linewidth=2.5,
        label=replace(main_labels[1], "STRICT " => ""),
    )

    # Plot strict BF16 RTN (behind remaining main curves, faint)
    h_overlay = fit(Histogram, overlay_counts, edges)
    probs_overlay = h_overlay.weights ./ sum(h_overlay.weights)
    stairs!(ax, centers .+ 1.5, probs_overlay;
        color=(PRECISION_COLORS[overlay_label], 0.25),
        linewidth=2.5,
        label="BF16 RTN",
    )

    # Plot remaining main curves on top
    linewidths = [2.0, 1.8]
    for (i, (lbl, counts)) in enumerate(zip(main_labels[2:end], main_counts[2:end]))
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        stairs!(ax, centers .+ 1.5, probs;
            color=PRECISION_COLORS[lbl],
            linewidth=linewidths[i],
            label=replace(lbl, "STRICT " => ""),
        )
    end

    Legend(fig[1, 1], ax;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=10,
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
