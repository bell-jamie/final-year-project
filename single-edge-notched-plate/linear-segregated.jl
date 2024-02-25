using Gridap
using GridapGmsh
using Gridap.TensorValues
using Gridap.Fields
using Gridap.CellData
using Gridap.Algebra
using Printf
using Plots
using TickTock
using CSV
using DataFrames
using Dates

include(joinpath(dirname(@__DIR__), "pfm-lib.jl"))

## Constants
const E = 210e3
const ν = 0.3
const C = elas_fourth_order_const_tensor(E, ν, "PlaneStrain")
const I4_vol, I4_dev = volumetric_deviatoric_projection()

const ls = 0.0075
const Gc = 2.7
const η = 1e-15

const max_cycles = 10
const tol = 1e-8
const δv_refined = 1e-5
const δv_coarse = 1e-4
const v_app_threshold = 5e-3
const v_app_max = 7e-3

## Model Setup
mesh_file = joinpath(@__DIR__, "notchedPlateTriangular.msh")
save_directory = create_save_directory(@__FILE__)
BCs = BoundaryConditions(["load", "fixed"], [(true, true), (true, true)], [2])
order = 2; degree = 2 * order

## Run
tick()
linear_segregated()
tock()

## Plot
plot_load_displacement("Single Edge Notched Plate - Linear Segregated")
plot_damage_displacement("Single Edge Notched Plate - Linear Segregated")
plot_increment_displacement("Single Edge Notched Plate - Linear Segregated")
plot_energy_displacement("Single Edge Notched Plate - Linear Segregated")