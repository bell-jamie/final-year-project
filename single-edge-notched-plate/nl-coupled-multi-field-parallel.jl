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
const nl_iter = 50
const verbose = true

## Displacement Parameters
const v_app_max = 6e-3
const linear_region = 5.3e-3
const δv_coarse = 1e-4
const δv_fine = 1e-5

## Model Setup
const mesh_file = joinpath(@__DIR__, "meshes", "notchedPlateTriangular.msh")
const save_directory = create_save_directory(@__FILE__)
const bc = BoundaryConditions(["load", "fixed"], [(false, true), (true, true)])
const order = 2
const degree = 2 * order

const threads = 4

## Run
tick()
with_mpi() do distribute
    non_linear_monolithic_parallel(distribute)
end
tock()

create_plots()

# Mac call with: mpiexecjl -n 6 julia nl-coupled-multi-field-parallel.jl
# Windows call with: mpiexec -n 6 julia nl-coupled-multi-field-parallel.jl