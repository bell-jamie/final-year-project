using GridapGmsh
using Gridap
using Gridap.TensorValues
using Gridap.Fields
using Gridap.CellData
using Printf
using PyPlot
using TickTock
using CSV
using DataFrames
using LineSearches: BackTracking
using Gridap.Algebra
using Dates

include(joinpath(dirname(@__DIR__), "pfm-lib.jl"))

const E = 210e3
const ν = 0.3
const C = elasFourthOrderConstTensor(E, ν, "PlaneStrain")
const I4_vol, I4_dev = volumetricDeviatoricProjection()

const ls = 0.0075
const gc_bulk = 2.7
const η = 1e-15

const growth_rate = 1.2
const max_cycles = 20
const tol = 1e-6
const δv_min = 1e-7
const δv_max = 1e-3
const v_app_max = 7e-3

## Model Setup
model = GmshDiscreteModel(joinpath(@__DIR__, "notchedPlateTriangular.msh"))
order = 2; degree = 2 * order

# Mesh Triangulation
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# FE Spaces
U_pf, V0_pf, sh = constructPhaseFieldSpace()
V0_disp, uh = constructDisplacementFieldSpace(["top", "bottom"], [(false, true), (true, true)])

Γ_load = BoundaryTriangulation(model, tags="top")
dΓ_load = Measure(Γ_load, degree)
n_Γ_load = get_normal_vector(Γ_load)

save_directory = createSaveDirectory(@__FILE__)

tick()
nonLinearRecursive()
tock()

#explicit types for calling the library functions to optimise and also allow multiple methods