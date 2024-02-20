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
const tol = 1e-5 #1e-6
const δv_min = 1e-7
const δv_max = 1e-5
const v_init = 2.5e-3
const v_app_max = 7e-3

## Model Setup
model = GmshDiscreteModel(joinpath(@__DIR__, "notchedPlateTriangular.msh"))
bcs = boundary_conditions(["top", "bottom"], [(true, true), (true, true)], [2])
order = 2; degree = 2 * order

# Mesh Triangulation
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# FE Spaces
U_pf, V0_pf, sh = constructPhaseFieldSpace()
V0_disp, uh = constructDisplacementFieldSpace(bcs.tags, bcs.masks)

Γ_load = BoundaryTriangulation(model, tags="top")
dΓ_load = Measure(Γ_load, degree)
n_Γ_load = get_normal_vector(Γ_load)

save_directory = createSaveDirectory(@__FILE__)

tick()
nonLinearCoupledRecursive()
tock()

plotLoadDisplacement("Single Edge Notched Plate - NL Coupled Recursive")

# let's plot the sum of the energy state with each iteration to see if the two methods are different