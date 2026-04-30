#!/usr/bin/env julia

"""
Generate Schlogl rescaled histogram for talk — solutions slide.

Shows 4 curves only: FP64 baseline, STRICT BF16 RTN (failure carried from slide 10),
STRICT BF16+SR (SR helps), STRICT FP16+SR (rescaling + SR solves it).

Usage:
  julia --project=. scripts/plot_schlogl_talk_solutions.jl
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
    output_stem = arg_string(opts, "output-stem", "figures/schlogl_talk_solutions")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # Five curves: reference, two RTN failures (faint), two SR fixes
    labels = [
        "FP64 baseline",
        "STRICT BF16 RTN",
        "STRICT FP16 RTN",
        "STRICT BF16 + SR",
        "STRICT FP16 + SR",
    ]

    # RTN labels drawn faint (matching slide 10 style)
    faint_labels = Set(["STRICT BF16 RTN", "STRICT FP16 RTN"])

    missing_labels = [lbl for lbl in labels if !haskey(ensemble, lbl)]
    isempty(missing_labels) || error("Missing labels in ensemble data: $(join(missing_labels, ", "))")

    counts_all = [ensemble[lbl]["counts"] for lbl in labels]

    lo = minimum(minimum.(counts_all))
    hi = maximum(maximum.(counts_all))
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

    linewidths = [2.5, 2.5, 2.5, 1.8, 1.8]
    for (i, (lbl, counts)) in enumerate(zip(labels, counts_all))
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        col = PRECISION_COLORS[lbl]
        alpha = lbl in faint_labels ? 0.25 : 1.0
        stairs!(ax, centers .+ 1.5, probs;
            color=(col, alpha),
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
