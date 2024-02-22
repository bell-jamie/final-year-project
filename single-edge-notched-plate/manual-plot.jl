using Gridap
using Printf
using CSV
using Plots

include(joinpath(dirname(@__DIR__), "pfm-lib.jl"))

save_directory = "single-edge-notched-plate/nl-coupled-multi-field-files/2024-02-21_01-18"

plot_load_displacement("Partial Plot")