#!/usr/bin/env julia

"""
Repressilator protein distribution — ALL 10 modes including strict RTN.
Comparison version; not the default figure.
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
    output_stem = arg_string(opts, "output-stem", "figures/repressilator_protein_dist_all")

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    ensemble = data["ensemble"]

    # All 10 modes
    labels = [
        "FP64 baseline",
        "FP32",
        "BF16 + SR",
        "FP16 + SR",
        "BF16 RTN",
        "FP16 RTN",
        "STRICT FP16 + SR",
        "STRICT BF16 + SR",
        "STRICT FP16 RTN",
        "STRICT BF16 RTN",
    ]

    # RTN strict modes drawn faint
    faint_labels = Set(["STRICT BF16 RTN", "STRICT FP16 RTN"])

    available = [lbl for lbl in labels if haskey(ensemble, lbl)]
    counts_all = [ensemble[lbl]["end_pA"] for lbl in available]

    lo = minimum(minimum.(counts_all))
    hi = maximum(maximum.(counts_all))
    edges = collect((lo - 2.5):5:(hi + 2.5))
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    set_theme!(theme_latexfonts())

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

    linewidths = [2.5, 2.0, 1.8, 1.5, 1.3, 1.0, 1.5, 2.0, 1.5, 2.0]
    for (i, (lbl, counts)) in enumerate(zip(available, counts_all))
        h = fit(Histogram, counts, edges)
        probs = h.weights ./ sum(h.weights)
        col = PRECISION_COLORS[lbl]
        alpha = lbl in faint_labels ? 0.25 : 1.0
        lw = i <= length(linewidths) ? linewidths[i] : 1.5
        stairs!(ax, centers .+ 2.5, probs;
            color=(col, alpha),
            linewidth=lw,
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
