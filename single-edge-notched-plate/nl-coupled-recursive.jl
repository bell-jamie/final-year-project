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
const δv_max = 1e-5
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
nonLinearRecursive()
tock()

# let's plot the sum of the energy state with each iteration to see if the two methods are different

savefile = CSV.File(joinpath(save_directory, "loadDisplacement.csv"))
displacement = savefile.Displacement
load = savefile.Force

plt.plot(displacement * 1e3, load)
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.title("Single Edge Notched Plate - Non-Linear Recursive")
plt.grid()
display(gcf())