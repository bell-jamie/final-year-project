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

const max_cycles = 100
const nl_tol = 1e-6
const s_tol = 1e-3

## Displacement Parameters
const v_init = 2.5e-3
const v_app_max = 6.5e-3
const v_app_threshold = 5e-3

## Displacement Adaptive Stepping
const δv_min = 1e-7
const δv_coarse_max = 1e-4
const δv_refine_max = 1e-5
const growth_rate = 1.2
const damage_criteria = false
const residual_criteria = true

## Model Setup
mesh_file = joinpath(@__DIR__, "meshes", "notchedPlateRahaman.msh")
save_directory = create_save_directory(@__FILE__)
bc = BoundaryConditions(["load", "fixed"], [(false, true), (true, true)], [2])
const order = 2
const degree = 2 * order

## Run
tick()
nl_coupled_recursive()
tock()

create_plots()
