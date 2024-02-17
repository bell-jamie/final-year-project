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

include("core-functions.jl") # this is going to cause problems with appyling the dirichlet conditions

## Constants
# Geometry Constants
const ls = 0.0075 # is this even needed? Nope! Just for the meshing file
const vb = 1

# Elasticity Constants
const E = 210e3 # 210 GPa - base units are mm, therefore MPa is the same as N/mm^2
const ν = 0.3
const C_mat = elasFourthOrderConstTensor(E, ν, "PlaneStrain")

# Fracture Constants
const gc_bulk = 2.7 # 2.7 N/mm
const η = 1e-15

# Time Step Constants
const growth_rate = 1.2 # δv can get larger if increment solved with no cutbacks
const recur_max = 20 # maximum number of recursive steps to minimise PDE residuals
const tol = 1e-6 # tolerance for PDE residuals
const δv_min = 1e-7 # minimum displacement increment
const δv_max = 1e-3 # maximum displacement increment (1e-5)
const v_app_max = 7e-3 # total applied displacement
# v_app = 0.1 * v_app_max # initial applied displacement

# Volumetric and Deviatoric Projection Tensors
I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
I4 = I2 ⊗ I2
I4_sym = one(SymFourthOrderTensorValue{2,Float64})
I4_vol = (1.0 / 2) * I4
I4_dev = I4_sym - I4_vol

## Model Setup
model = GmshDiscreteModel(joinpath(@__DIR__, "notchedPlateTriangular.msh"))
order = 2
degree = 2 * order

# Mesh Triangulation
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
reffe = ReferenceFE(lagrangian, Float64, order)
V = FESpace(model, reffe, conformity=:L2)

# Phase Field
reffe_pf = ReferenceFE(lagrangian, Float64, order)
V0_pf = TestFESpace(model, reffe_pf, conformity=:H1)
U_pf = TrialFESpace(V0_pf)
sh = FEFunction(V0_pf, ones(num_free_dofs(V0_pf)))

# Displacement Field
reffe_disp = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V0_disp = TestFESpace(model, reffe_disp, conformity=:H1,
    dirichlet_tags=["top", "bottom"],
    dirichlet_masks=[(false, true), (true, true)])
uh = zero(V0_disp)

# Apply Load
Γ_load = BoundaryTriangulation(model, tags="top")
dΓ_load = Measure(Γ_load, degree)
n_Γ_load = get_normal_vector(Γ_load)

# Initialise Parameters
count = 1
δv = δv_max # initial displacement increment
v_app = 0; # initial applied displacement

load = Float64[]
push!(load, 0.0)

displacement = Float64[]
push!(displacement, 0.0)

ψ_plus_prev = CellState(0.0, dΩ)

### Save Directory ###
date = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")
save_directory = joinpath(splitext(@__FILE__)[1] * "-files", date) # make this global

if !isdir(save_directory)
    mkpath(save_directory)
end

### Start Timer ###
tick()


## Main Loop
while v_app .< v_app_max
    global v_app += δv # apply increment
    
    @printf("\n%s\n", centre_pad(" Step: $count ", 40))
    @printf("%s\n\n", centre_pad(" Displacement: $v_app m ", 40)) # i need to develop a way to format this in scientific notation with padding
    # maybe standalone print function that handles the padding? so it could take the same args as printf + the width and would directly call printf

    converged = false
    cut_back = false
    early_quit = false

    for recur ∈ 1:recur_max
        @printf("%s\n", centre_pad(" Cycle: $recur ", 40))
        @printf("%s\n\n", centre_pad(" Increment: $δv m ", 40)) # why are we printing increment each time?
        # maybe print residual tolerance?

        # Solve Phase Field
        @printf("%s\n", centre_pad(" Solving Phase-Field ", 40))
        global sh, pf_residual = stepPhaseField(sh, ψ_plus_prev, tol)
        @printf("%s\n\n", centre_pad(" Residual: $pf_residual", 40)) # i would like to round the residual at some point

        # Solve Displacement Field
        @printf("%s\n", centre_pad(" Solving Displacement Field ", 40))
        global uh, disp_residual = stepDisplacement(uh, sh, v_app, tol)
        @printf("%s\n\n", centre_pad(" Residual: $disp_residual", 40))

        # Update Energy State
        ψ_pos_in = ψPos ∘ (ε(uh))
        update_state!(newEnergyState, ψ_plus_prev, ψ_pos_in)

        # Check for convergence
        if pf_residual < tol && disp_residual < tol
            converged = true
            @printf("PDE residuals converged\n\n")
            break
        end

        # Check for cutback
        # I think that an adaptive cutback could be implemented, based on the magnitude of the residual, saving the number of cycles to cut back...
        if recur == recur_max
            @printf("Max recursive steps reached - cutting back\n\n")
            cut_back = true
            v_app -= δv # remove increment
            δv = δv / 2 # halve increment
            # should probably pop() the last energy state otherwise there will be permanent damage not accounted for
            if δv < δv_min
                @printf("Minimum displacement increment reached - early quit\n\n")
                early_quit = true
                break
            end
        end
    end

    if converged
        node_force = sum(∫(n_Γ_load ⋅ (σMod ∘ (ε(uh), ε(uh), sh))) * dΓ_load) * vb
        #yell = push!(load, node_force[2]) # why yell?
        push!(load, node_force[2])
        push!(displacement, v_app)

        if mod(count, 10) == 0 # write every 10th iteration
            writevtk(Ω, joinpath(save_directory, "partialSolve.vtu"),
                cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])
        end

        if v_app >= 0 && mod(count, 10) == 0 # displacement has been applied and every 10th iteration
            data_frame = DataFrame(Displacement=displacement, Force=load)
            CSV.write(joinpath(save_directory, "partialSolve.csv"), data_frame)
        end

        global δv = min(δv * growth_rate, δv_max)
        global count += 1
    end

    if early_quit
        break
    end
end

tock()


## Results
# Write File
writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
    cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])

# Plotting
plt.plot(displacement * 1e3, load)
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
# plt.legend("")
plt.title("Single Edge Notched Plate - Non-Linear Recursive")
plt.grid()
display(gcf())

data_frame = DataFrame(Displacement=displacement, Force=load)
CSV.write(joinpath(save_directory, "fullSolve.csv"), data_frame)