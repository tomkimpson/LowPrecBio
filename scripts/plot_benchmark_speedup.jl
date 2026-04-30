#!/usr/bin/env julia

"""
Generate publication-quality benchmark speedup figure using CairoMakie.

Grouped bar chart showing throughput speedup relative to FP64 baseline
for each model under mixed-precision mode. Schlogl excluded due to
anomalous memory-allocation behaviour.

Usage:
  julia --project=. scripts/plot_benchmark_speedup.jl
  julia --project=. scripts/plot_benchmark_speedup.jl \
    --input-csv=results/benchmarks/summary.csv \
    --output-stem=figures/benchmark_speedup
"""

using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

function read_csv_rows(path::String)
    lines = readlines(path)
    isempty(lines) && return Dict{String, String}[]
    header = split(lines[1], ",")
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ","; keepempty=true)
        row = Dict{String, String}()
        for i in eachindex(header)
            row[header[i]] = i <= length(vals) ? vals[i] : ""
        end
        push!(rows, row)
    end
    return rows
end

function main()
    opts = parse_cli_args(ARGS)
    input_csv = arg_string(opts, "input-csv", "results/benchmarks/summary.csv")
    output_stem = arg_string(opts, "output-stem", "figures/benchmark_speedup")

    isfile(input_csv) || error("Input CSV not found: $input_csv")
    rows = read_csv_rows(input_csv)
    isempty(rows) && error("No rows in $input_csv")

    # Schlogl excluded (anomalous memory-allocation artefacts)
    model_order = ["birth_death", "telegraph", "dimer", "repressilator"]
    label_order = ["FP32", "BF16 + SR", "FP16 + SR", "BF16 RTN", "FP16 RTN"]

    # model => label => events/s
    events = Dict{String, Dict{String, Float64}}()
    for r in rows
        model = r["model"]
        label = r["label"]
        ev = tryparse(Float64, r["events_per_second"])
        ev === nothing && continue
        if !haskey(events, model)
            events[model] = Dict{String, Float64}()
        end
        events[model][label] = ev
    end

    # Keep only models present in the benchmark CSV
    models = [m for m in model_order if haskey(events, m) && haskey(events[m], "FP64 baseline")]
    isempty(models) && error("No FP64 baseline rows found in $input_csv")

    n_models = length(models)
    n_series = length(label_order)

    speedup = Matrix{Float64}(undef, n_models, n_series)
    for (i, model) in enumerate(models)
        baseline = events[model]["FP64 baseline"]
        for (j, label) in enumerate(label_order)
            haskey(events[model], label) || error("Missing row for model=$model label=$label")
            speedup[i, j] = events[model][label] / baseline
        end
    end

    model_names = Dict(
        "birth_death" => "Birth–Death",
        "telegraph" => "Telegraph",
        "dimer" => "Dimer",
        "repressilator" => "Repressilator",
    )
    model_display = [get(model_names, m, m) for m in models]

    # Flatten for CairoMakie barplot with dodge
    x_flat = repeat(1:n_models, outer=n_series)
    dodge_flat = repeat(1:n_series, inner=n_models)
    vals_flat = vec(speedup)  # column-major: all models for series 1, then series 2, ...

    # Map label_order to PRECISION_COLORS
    series_colors = [PRECISION_COLORS[lbl] for lbl in label_order]
    color_flat = [series_colors[d] for d in dodge_flat]

    # --- CairoMakie figure ---
    set_theme!(theme_latexfonts())

    fig = Figure(size=(700, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel="Model",
        ylabel="Speedup vs FP64",
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        xticks=(1:n_models, model_display),
        limits=((0.3, n_models + 1.2), nothing),
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    barplot!(ax, x_flat, vals_flat;
        dodge=dodge_flat,
        color=color_flat,
    )

    # FP64 parity line
    hlines!(ax, [1.0]; color=:gray40, linewidth=1.0, linestyle=:dash)

    # Legend entries (manual, since dodge doesn't auto-generate legend)
    legend_entries = Vector{LegendElement}[
        [PolyElement(; color=series_colors[j])] for j in 1:n_series
    ]
    legend_labels = copy(label_order)
    push!(legend_entries, [LineElement(; color=:gray40, linestyle=:dash, linewidth=1.0)])
    push!(legend_labels, "FP64 parity")

    Legend(fig[1, 1], legend_entries, legend_labels;
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
