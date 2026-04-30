#!/usr/bin/env julia

"""
Generate publication-quality repressilator trajectory figure using CairoMakie.

Runs a single FP64 and BF16+SR trajectory and overlays them.

Usage:
  julia --project=. scripts/plot_repressilator_trajectory.jl
"""

using LowPrecBio
using StochasticRounding
using CairoMakie
using LaTeXStrings

include("cli_args.jl")
include("precision_colors.jl")

function main()
    opts = parse_cli_args(ARGS)
    output_stem = arg_string(opts, "output-stem", "figures/repressilator_trajectory")
    seed_ssa = arg_int(opts, "seed-ssa", 123)
    seed_sr = arg_int(opts, "seed-sr", 3030)
    t_end = arg_float(opts, "t-end", 200.0)

    model = RepressilatorModel(
        alpha0=1.0, alpha=216.0, n=2,
        delta_m=1.0, beta=5.0, delta_p=1.0,
        initial_pA=5,
    )

    # FP64 trajectory
    dual_fp64 = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual_fp64)
    traj_fp64 = ssa_repressilator!(model, t_end;
        Tprop=Float64, Tacc=Float64, rng=dual_fp64.ssa, mode=:mixed)

    # BF16+SR trajectory (same seeds)
    dual_bfsr = DualRNG(seed_ssa, seed_sr)
    activate_sr_rng!(dual_bfsr)
    traj_bfsr = ssa_repressilator!(model, t_end;
        Tprop=BFloat16sr, Tacc=Float32, rng=dual_bfsr.ssa, mode=:mixed)

    set_theme!(theme_latexfonts())

    fig = Figure(size=(468, 280), figure_padding=(2, 8, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel="Time",
        ylabel=L"Protein A count, $n_{pA}$",
        xlabelsize=14,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12,
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )

    lines!(ax, traj_fp64.times, traj_fp64.counts_pA;
        color=(PRECISION_COLORS["FP64 baseline"], 0.7),
        linewidth=1.0,
        label="FP64 baseline",
    )
    lines!(ax, traj_bfsr.times, traj_bfsr.counts_pA;
        color=(PRECISION_COLORS["BF16 + SR"], 0.7),
        linewidth=1.0,
        linestyle=:dash,
        label="BF16 + SR",
    )

    Legend(fig[1, 1], ax;
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
