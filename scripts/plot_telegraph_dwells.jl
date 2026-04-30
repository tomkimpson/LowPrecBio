#!/usr/bin/env julia

"""
Generate publication-quality telegraph dwell-time distribution figure using CairoMakie.

Two-panel figure: OFF dwell times (left), ON dwell times (right).
Shows FP64 baseline + BF16+SR with analytical exponential PDF overlay.

Usage:
  julia --project=. scripts/plot_telegraph_dwells.jl
  julia --project=. scripts/plot_telegraph_dwells.jl \
    --input-jld2=results/telegraph/ensemble_data.jld2 \
    --metadata-jld2=results/telegraph/metadata.jld2 \
    --output-stem=figures/telegraph_dwells
"""

using JLD2
using StatsBase
using CairoMakie
using LaTeXStrings

include("cli_args.jl")

function main()
    opts = parse_cli_args(ARGS)
    input_jld2 = arg_string(opts, "input-jld2", "results/telegraph/ensemble_data.jld2")
    metadata_jld2 = arg_string(opts, "metadata-jld2", "results/telegraph/metadata.jld2")
    output_stem = arg_string(opts, "output-stem", "figures/telegraph_dwells")
    k_on_default = arg_float(opts, "k-on", 0.01)
    k_off_default = arg_float(opts, "k-off", 0.1)

    isfile(input_jld2) || error("Input file not found: $input_jld2")

    data = JLD2.load(input_jld2)
    haskey(data, "ensemble") || error("Expected key 'ensemble' in $input_jld2")
    ensemble = data["ensemble"]

    # Load k_on/k_off from metadata if available, otherwise use CLI defaults
    k_on = k_on_default
    k_off = k_off_default
    if isfile(metadata_jld2)
        meta = JLD2.load(metadata_jld2, "metadata")
        if haskey(meta, "model_params")
            params = meta["model_params"]
            k_on = get(params, "k_on", k_on_default)
            k_off = get(params, "k_off", k_off_default)
        end
    end

    labels = ["FP64 baseline", "BF16 + SR"]
    missing_labels = [lbl for lbl in labels if !haskey(ensemble, lbl)]
    isempty(missing_labels) || error("Missing labels in ensemble data: $(join(missing_labels, ", "))")

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    colors = Makie.wong_colors()

    fig = Figure(size=(700, 280), figure_padding=(2, 8, 2, 2))

    # ----- Left panel: OFF dwell times -----
    ax_off = Axis(fig[1, 1],
        xlabel="Dwell time",
        ylabel="Density",
        title="OFF state",
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

    for (i, lbl) in enumerate(labels)
        dwells = ensemble[lbl]["off_dwells"]
        length(dwells) > 0 || continue
        h = fit(Histogram, dwells; nbins=50)
        total = sum(h.weights)
        widths = diff(h.edges[1])
        probs = h.weights ./ (total .* widths)
        centers = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2
        stairs!(ax_off, collect(h.edges[1][2:end]), probs;
            color=colors[i],
            linewidth=i == 1 ? 2.0 : 1.5,
            label=lbl,
        )
    end

    # Analytical exponential PDF: rate = k_on for OFF dwell times
    max_off = maximum(ensemble["FP64 baseline"]["off_dwells"])
    ts = range(0, max_off; length=200)
    lines!(ax_off, collect(ts), k_on .* exp.(-k_on .* ts);
        color=:black,
        linewidth=1.5,
        linestyle=:dash,
        label=L"Exp($k_\mathrm{on}$)",
    )

    Legend(fig[1, 1], ax_off;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=10,
        framevisible=false,
    )

    # ----- Right panel: ON dwell times -----
    ax_on = Axis(fig[1, 2],
        xlabel="Dwell time",
        ylabel="Density",
        title="ON state",
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

    for (i, lbl) in enumerate(labels)
        dwells = ensemble[lbl]["on_dwells"]
        length(dwells) > 0 || continue
        h = fit(Histogram, dwells; nbins=50)
        total = sum(h.weights)
        widths = diff(h.edges[1])
        probs = h.weights ./ (total .* widths)
        centers = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2
        stairs!(ax_on, collect(h.edges[1][2:end]), probs;
            color=colors[i],
            linewidth=i == 1 ? 2.0 : 1.5,
            label=lbl,
        )
    end

    # Analytical exponential PDF: rate = k_off for ON dwell times
    max_on = maximum(ensemble["FP64 baseline"]["on_dwells"])
    ts_on = range(0, max_on; length=200)
    lines!(ax_on, collect(ts_on), k_off .* exp.(-k_off .* ts_on);
        color=:black,
        linewidth=1.5,
        linestyle=:dash,
        label=L"Exp($k_\mathrm{off}$)",
    )

    Legend(fig[1, 2], ax_on;
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
        margin=(8, 8, 8, 8),
        padding=(4, 4, 3, 3),
        labelsize=10,
        framevisible=false,
    )

    mkpath(dirname(output_stem))
    save("$(output_stem).pdf", fig)
    save("$(output_stem).png", fig; px_per_unit=4)
    println("Saved: $(output_stem).pdf")
    println("Saved: $(output_stem).png")
end

main()
