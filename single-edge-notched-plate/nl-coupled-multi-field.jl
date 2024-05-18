using Gridap
using GridapGmsh
using Gridap.TensorValues
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
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
const ls = 0.0075
const Gc = 2.7
const η = 1e-15
const nl_tol = 1e-5
const nl_iter = 1000
const verbose = true

## Displacement Parameters
const v_app_max = 6e-3
const linear_region = 5.3e-3
const δv_coarse = 1e-4
const δv_fine = 1e-6

## Model Setup
const mesh_file = joinpath(@__DIR__, "meshes", "notchedPlateRahaman.msh")
const save_directory = create_save_directory(@__FILE__)
const bc = BoundaryConditions(["load", "fixed"], [(false, true), (true, true)])
const order = 2
const degree = 2 * order

## Run
tick()
non_linear_monolithic()
tock()

create_plots()
