using GridapGmsh
using Gridap
#using Gridap.Geometry
using Gridap.TensorValues
using Gridap.Fields
#using FillArrays
using Gridap.CellData
using Printf
using PyPlot
#using PyCall
using TickTock
using CSV
using DataFrames
using LineSearches: BackTracking
#using NLsolve
#using ForwardDiff
using Gridap.Algebra # for the residual function


## Functions
function elasFourthOrderConstTensor(E, ν, PlanarState)
    if PlanarState == "PlaneStrain"
        C1111 = E / (1 - ν * ν)
        C1122 = (E * ν) / (1 - ν * ν)
        C1112 = 0.0
        C2222 = E / (1 - ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    elseif PlanarState == "PlaneStress"
        C1111 = (E * (1 - ν * ν)) / ((1 + ν) * (1 - ν - 2 * ν * ν))
        C1122 = (E * ν) / (1 - ν - 2 * ν * ν)
        C1112 = 0.0
        C2222 = (E * (1 - ν)) / (1 - ν - 2 * ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    else
        error("Invalid PlanarState")
    end
    SymFourthOrderTensorValue(C1111, C1112, C1122, C1112, C1212, C2212, C1122, C2212, C2222)
end

function σMod(ε, ε_in, s_in)
    if tr(ε_in) >= 0
        (s_in^2 + η) * σ(ε)
    else
        (s_in^2 + η) * I4_dev ⊙ σ(ε) + I4_vol ⊙ σ(ε)
    end
end

function ψPos(ε_in)
    if tr(ε_in) >= 0
        0.5 * (ε_in ⊙ σ(ε_in))
    else
        0.5 * (I4_dev ⊙ σ(ε_in) ⊙ (I4_dev ⊙ ε_in))
    end
end
#=
function newEnergyState(ψ_plus_prev_in, ψ_plus_in)
    if ψ_plus_in >= ψ_plus_prev_in
        ψ_plus_out = ψ_plus_in
    else
        ψ_plus_out = ψ_plus_prev_in
    end
    true, ψ_plus_out # why true?
end
=#
function newEnergyState(ψ_plus_prev_in, ψ_plus_in)
    if ψ_plus_in >= ψ_plus_prev_in
        ψ_plus_in
    else
        ψ_plus_prev_in
    end
end

function stepPhaseField(sh_in, ψ_plus_prev_in, f_tol)
    #res_pf(s, ϕ) = ∫(Gc(tags) * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_plus_prev_in * s * ϕ + (Gc(tags) / ls) * s * ϕ) * dΩ - ∫((Gc(tags) * ϕ) / ls) * dΩ
    res_pf(s, ϕ) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_plus_prev_in * s * ϕ + (gc_bulk / ls) * s * ϕ) * dΩ - ∫((gc_bulk * ϕ) / ls) * dΩ
    #jac_pf(s, ds, ϕ) = ∫(Gc(tags) * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψ_plus_prev_in * ds * ϕ + (Gc(tags) / ls) * ds * ϕ) * dΩ
    jac_pf(s, ds, ϕ) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψ_plus_prev_in * ds * ϕ + (gc_bulk / ls) * ds * ϕ) * dΩ
    op_pf = FEOperator(res_pf, jac_pf, U_pf, V0_pf)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), ftol=f_tol, iterations=5)
    solver_pf = FESolver(nls)
    sh_out, = solve!(sh_in, solver_pf, op_pf) # don't remove "," - turns out this was very needed!
    return sh_out, norm(residual(op_pf, sh_out), Inf)
end

function stepDisplacement(uh_in, sh_in, v_app, f_tol)
    u_app_1 = VectorValue(0.0, v_app)
    u_app_2 = VectorValue(0.0, 0.0)
    U_disp = TrialFESpace(V0_disp, [u_app_1, u_app_2])
    res_disp(u, v) = ∫(ε(v) ⊙ (σMod ∘ (ε(u), ε(uh_in), sh_in))) * dΩ
    jac_disp(u, du, v) = ∫(ε(v) ⊙ (σMod ∘ (ε(du), ε(uh_in), sh_in))) * dΩ
    op_disp = FEOperator(res_disp, jac_disp, U_disp, V0_disp)
    nls_disp = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), ftol=f_tol, iterations=5)
    solver_disp = FESolver(nls_disp) # in theory this could be nested directly in solve! and nls_disp inside FESolver
    uh_out, = solve!(uh_in, solver_disp, op_disp)
    return uh_out, norm(residual(op_disp, uh_out), Inf)
end
#=
function Gc(tag) # this function may not be required for single edge notch
    Gc_by_tag = Float64[]
    for i ∈ eachindex(tag)
        if tag[i] == CrackLine
            push!(Gc_by_tag, gc)
        else
            push!(Gc_by_tag, 0.0)
        end
    end # no explicit return required
end

function project(f, V, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    l(v) = ∫(v * f) * dΩ
    op = AffineFEOperator(a, l, V, V)
    solve(op)
end
=#
function σ(ε)
    C_mat ⊙ ε
end

#σ(ε) = C_mat ⊙ ε


## Constants
# Geometry Constants
const ls = 3e-5
const L = 100e-3
const CL = 0.3 * L
const vb = 1
const H = 4e-3 + 1e-4
const b = 5e-5

# Elasticity Constants
const E = 126e9
const ν = 0.3
const C_mat = elasFourthOrderConstTensor(E, ν, "PlaneStrain")

# Fracture Constants
const gc_bulk = 187
const η = 1e-8

# Time Step Constants
const growth_rate = 1.2 # δv can get larger if increment solved with no cutbacks
const recur_max = 20 # maximum number of recursive steps to minimise PDE residuals
const tol = 1e-6 # tolerance for PDE residuals
const δv_min = 1e-7 # minimum displacement increment
const δv_max = 1e-5 # maximum displacement increment
const v_app_max = 2e-3 # total applied displacement
# v_app = 0.1 * v_app_max # initial applied displacement

# Volumetric and Deviatoric Projection Tensors
I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
I4 = I2 ⊗ I2
I4_sym = one(SymFourthOrderTensorValue{2,Float64})


## Model Setup
model = GmshDiscreteModel("notchedPlateTriangular.msh")

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
V0_disp = TestFESpace(model, reffe_disp, conformity=:H1, dirichlet_tags=["top", "bottom"], dirichlet_masks=[(true, true), (true, true)])
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

tick()


## Main Loop
while v_app .< v_app_max
    global v_app += δv # apply increment
    @printf("Displacement step: %i // Applied displacement: %.7e\n", count, float(v_app))

    converged = false
    cut_back = false
    early_quit = false

    for recur ∈ 1:recur_max
        @printf("Recursive step: %i // Displacement increment: %.7e\n", recur, δv)

        # Solve Phase Field
        global sh, pf_residual = stepPhaseField(sh, ψ_plus_prev, tol)
        @printf("Phase field residual: %.6e\n", pf_residual)

        # Solve Displacement Field
        global uh, disp_residual = stepDisplacement(uh, sh, v_app, tol)
        @printf("Displacement residual: %.6e\n", disp_residual)

        ψ_pos_in = ψPos ∘ (ε(uh))
        update_state!(newEnergyState, ψ_plus_prev, ψ_pos_in)

        # Check for convergence
        if pf_residual < tol && disp_residual < tol
            converged = true
            @printf("PDE residuals converged\n")
            break
        else
            @printf("PDE residuals not converged\n")
        end

        # Check for cutback
        if recur == recur_max
            @printf("Max recursive steps reached - cutting back\n")
            cut_back = true
            v_app -= δv # remove increment
            δv = δv / 2 # halve increment
            if δv < δv_min
                @printf("Minimum displacement increment reached - early quit\n")
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

        if mod(count, 20) == 0 # write every 20th iteration
            writevtk(Ω, "partialSolve-iter-$count.vtu", cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])
        end

        if v_app >= 0 && mod(count, 10) == 0 # displacement has been applied and every 10th iteration
            data_frame = DataFrame(Displacement=displacement, Force=load)
            CSV.write("partialSolve-iter-$count.csv", data_frame) # rewrite to create only one continuining csv
        end

        global δv = min(δv * growth_rate, δv_max)
        global count += 1
    end

    if early_quit
        break
    end
end

tock() # end timer


## Results
# Write File
writevtk(Ω, "SENPCoupledRecursive.vtu", cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])

# Plotting
plt.plot(displacement * 1e3, load)
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
# plt.legend("")
plt.title("Single Edge Notched Plate - Non-Linear Recursive")
plt.grid()
plt.show()

data_frame = DataFrame(Displacement=displacement, Force=load)
CSV.write("fullSolve.csv", data_frame)