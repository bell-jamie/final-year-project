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
using LineSearches: BackTracking
using Dates

include(joinpath("..", "pfm-lib.jl"))

## Constants
const E = 210e3
const ν = 0.3
const C = elas_fourth_order_const_tensor(E, ν, "PlaneStrain")
const I4_vol, I4_dev = volumetric_deviatoric_projection()

const ls = 0.0075
const Gc = 2.7
const η = 1e-15

const growth_rate = 1.2
const max_cycles = 20
const tol = 1e-6
const δv_min = 1e-8 # 1e-7
const δv_max = 1e-4 # 1e-5
const v_init = 2.5e-3
const v_app_max = 7e-3

## Displacement Adaptive Stepping
const δv_coarse = 1e-4
const δv_refined = 1e-5
const v_app_threshold = 5e-3

## Model Setup
mesh_file = joinpath(@__DIR__, "meshes", "notchedPlateRahaman.msh")
save_directory = create_save_directory(@__FILE__)
BCs = BoundaryConditions(["load", "fixed"], [(true, true), (true, true)], [2])
const order = 2
const degree = 2 * order

## Run
tick()
NL_coupled_recursive()
tock()

## Plot
plot_load_displacement("Single Edge Notched Plate - NL Coupled Recursive")
plot_damage_displacement("Single Edge Notched Plate - NL Coupled Recursive")
plot_increment_displacement("Single Edge Notched Plate - NL Coupled Recursive")
plot_energy_displacement("Single Edge Notched Plate - NL Coupled Recursive")