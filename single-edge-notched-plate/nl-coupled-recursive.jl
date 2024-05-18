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
const ℂ = stiffness_tensor(E, ν, "PlaneStrain")
const Iᵛᵒˡ, Iᵈᵉᵛ = vol_dev()
const ls = 0.0075 # 0.0075 - 0.015 - 0.03
const Gc = 2.7
const η = 1e-15
const nl_tol = 1e-5
const nl_iter = 5
const verbose = false

## Displacement Parameters
const v_app_max = 6e-3
const linear_region = 5e-3
const δv_coarse = 1e-4
const δv_fine = 15e-5

## Model Setup
const mesh_file = joinpath(@__DIR__, "meshes", "notchedPlateRahaman.msh")
const save_directory = create_save_directory(@__FILE__)
const bc = BoundaryConditions(["load", "fixed"], [(false, true), (true, true)])
const order = 2
const degree = 2 * order

## Run
tick()
non_linear_alternate_minimisation()
tock()

create_plots()
