# precision_colors.jl — consistent color assignments across all precision-mode plots.
# Uses Makie's Wong colorblind-safe palette. Mixed and strict variants of the
# same format+rounding share a color; panel labels distinguish the architecture.
#
# Requires CairoMakie (or Makie) to be loaded before including this file.

const PRECISION_COLORS = let c = Makie.wong_colors()
    Dict(
        "FP64 baseline"    => c[1],  # dark blue
        "FP32"             => c[2],  # orange
        "BF16 + SR"        => c[3],  # teal
        "STRICT BF16 + SR" => c[3],
        "FP16 + SR"        => c[4],  # pink
        "STRICT FP16 + SR" => c[4],
        "BF16 RTN"         => c[5],  # sky blue
        "STRICT BF16 RTN"  => c[5],
        "FP16 RTN"         => c[7],  # vermillion (skip yellow c[6])
        "STRICT FP16 RTN"  => c[7],
    )
end
