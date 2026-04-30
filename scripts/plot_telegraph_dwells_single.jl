#!/usr/bin/env julia

"""
Generate single-panel telegraph dwell-time distribution figure using CairoMakie.

Shows OFF dwell times only (ON is identical when k_on = k_off).
FP64 baseline + BF16+SR with analytical exponential PDF overlay.

Usage:
  julia --project=. scripts/plot_telegraph_dwells_single.jl \
    --input-jld2=results/telegraph_bimodal/ensemble_data.jld2 \
    --metadata-jld2=results/telegraph_bimodal/metadata.jld2 \
    --output-stem=figures/telegraph_bimodal_dwells_single
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/telegraph_bimodal/ensemble_data.jld2")
    metadata_jld2 = arg_string(opts, "metadata-jld2", "results/telegraph_bimodal/metadata.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/telegraph_bimodal_dwells_single")
    k_on_default = arg_float(opts, "k-on", 0.05)

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # Load k_on from metadata if available
    k_on = k_on_default
    if isfile(metadata_jld2)
        meta = JLD2.load(metadata_jld2, "metadata")
        if haskey(meta, "model_params")
            params = meta["model_params"]
            k_on = get(params, "k_on", k_on_default)
        end
    end

    labels = ["FP64 baseline", "BF16 + SR"]
    missing_labels = [lbl for lbl in labels if !haskey(ensemble, lbl)]
    isempty(missing_labels) || error("Missing labels in ensemble data: $(join(missing_labels, ", "))")

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    colors = Makie.wong_colors()

    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel="Dwell time",
        ylabel="Density",
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    for (i, lbl) in enumerate(labels)
        dwells = ensemble[lbl]["off_dwells"]
        length(dwells) > 0 || continue
        h = fit(Histogram, dwells; nbins=50)
        total = sum(h.weights)
        widths = diff(h.edges[1])
        probs = h.weights ./ (total .* widths)
        stairs!(ax, collect(h.edges[1][2:end]), probs;
            color=colors[i],
            linewidth=i == 1 ? 2.0 : 1.5,
            label=lbl,
        )
    end

    # Analytical exponential PDF: rate = k_on for OFF dwell times
    max_off = maximum(ensemble["FP64 baseline"]["off_dwells"])
    ts = range(0, max_off; length=200)
    lines!(ax, collect(ts), k_on .* exp.(-k_on .* ts);
        color=:black,
        linewidth=1.5,
        linestyle=:dash,
        label=L"Exp($k_\mathrm{on}$)",
    )

    Legend(fig[1, 1], ax;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=12,
        framevisible=false,
    )

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
