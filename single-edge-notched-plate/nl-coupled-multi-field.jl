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

include(joinpath(dirname(@__DIR__), "pfm-lib.jl"))

## Constants
const E = 210e3
const ν = 0.3
const C = elas_fourth_order_const_tensor(E, ν, "PlaneStrain")
const I4_vol, I4_dev = volumetric_deviatoric_projection()

const ls = 0.0075
const gc_bulk = 2.7
const η = 1e-15

const growth_rate = 1.2
const NL_iters = 20
const tol = 2e-12
const δv_min = 1e-10
const δv_max = 1e-5
const v_init = 2.5e-3
const v_app_max = 15e-3

## Model Setup
mesh_file = joinpath(@__DIR__, "notchedPlateTriangular.msh")
save_directory = create_save_directory(@__FILE__)
BCs = boundary_conditions(["load", "fixed"], [(true, true), (true, true)], [2])
order = 2; degree = 2 * order

## Run
tick()
NL_coupled_multi_field()
tock()

## Plot
plot_load_displacement("Single Edge Notched Plate - NL Coupled Multi-Field")