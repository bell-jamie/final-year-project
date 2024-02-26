using Gridap
using GridapDistributed
using GridapPETSc
using GridapGmsh
using Gridap.TensorValues
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.Algebra
using PartitionedArrays
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
const Gc = 2.7
const η = 1e-15

const growth_rate = 1.2
const NL_iters = 20
const tol = 2e-12 # 1e-12
const δv_min = 1e-7 # 1e-7
const δv_max = 1e-4 # 1e-5
const v_init = 2.5e-3
const v_app_max = 7e-3

## Model Setup
mesh_file = joinpath(@__DIR__, "notchedPlateTriangular.msh")
save_directory = create_save_directory(@__FILE__)
BCs = BoundaryConditions(["load", "fixed"], [(true, true), (true, true)], [2])
const order = 2
const degree = 2 * order

# Mac call with: mpiexecjl -n 6 julia nl-coupled-multi-field-parallel.jl
# Windows call with: mpiexec -n 6 julia nl-coupled-multi-field-parallel.jl

## Run
tick()
with_mpi() do distribute
    ranks = distribute_with_mpi(LinearIndices((6,)))
    NL_coupled_multi_field_parallel(ranks)
end
tock()

## Plot
plot_load_displacement("Single Edge Notched Plate - NL Coupled Multi-Field")
plot_damage_displacement("Single Edge Notched Plate - NL Coupled Multi-Field")
plot_increment_displacement("Single Edge Notched Plate - NL Coupled Multi-Field")
plot_energy_displacement("Single Edge Notched Plate - NL Coupled Multi-Field")