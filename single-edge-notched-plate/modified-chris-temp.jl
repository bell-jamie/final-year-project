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
println("1")
const ls = 3e-5
const L = 100e-3
const CL = 0.3 * L
const vb = 1 #30e-3 #virtual width
const H = 4e-3 + 1e-4
const b = 5e-5
model = GmshDiscreteModel("notchedPlateTriangular.msh")

I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
I4 = I2 ⊗ I2
I4_sym = one(SymFourthOrderTensorValue{2,Float64})
I4_vol = (1.0 / 2) * I4
I4_dev = I4_sym - I4_vol

#Elasticity Paramters                                    
const E_mat = 126e9
const ν_mat = 0.3

#Fracture Parameters
const G_cprev = 281
const G_cbulk = G_cprev * 10
const G_c = 187.44291603380015
# const G_c = (G_cprev - G_cbulk*exp(-(b/(ls/2)))) / (1 - exp(-b/(ls/2)))
const η = 1e-8
println(G_c)
println(b / (ls / 2))
# println(log(G_cbulk/G_cprev))

#Constitutive Tensor
function ElasFourthOrderConstTensor(E, ν, PlanarState)
    # 1 for  Plane  Stress  and 2 Plane  Strain  Condition
    if PlanarState == 1
        C1111 = E / (1 - ν * ν)
        C1122 = (ν * E) / (1 - ν * ν)
        C1112 = 0.0
        C2222 = E / (1 - ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    elseif PlanarState == 2
        C1111 = (E * (1 - ν * ν)) / ((1 + ν) * (1 - ν - 2 * ν * ν))
        C1122 = (ν * E) / (1 - ν - 2 * ν * ν)
        C1112 = 0.0
        C2222 = (E * (1 - ν)) / (1 - ν - 2 * ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    end
    C_ten = SymFourthOrderTensorValue(C1111, C1112, C1122, C1112, C1212, C2212, C1122, C2212, C2222)
    return C_ten
end
const C_mat = ElasFourthOrderConstTensor(E_mat, ν_mat, 1)

#Stress
σ_elas(ε) = C_mat ⊙ ε

function σ_mod(ε, ε_in, s_in)
    if tr(ε_in) >= 0
        σ = (s_in^2 + η) * σ_elas(ε)
    elseif tr(ε_in) < 0
        σ = (s_in^2 + η) * I4_dev ⊙ σ_elas(ε) + I4_vol ⊙ σ_elas(ε)
    end
    return σ
end

#Elastic Strain energy
function ψPos(ε_in)
    if tr(ε_in) >= 0
        ψPlus = 0.5 * (ε_in ⊙ σ_elas(ε_in))
    elseif tr(ε_in) < 0
        ψPlus = 0.5 * ((I4_dev ⊙ σ_elas(ε_in)) ⊙ (I4_dev ⊙ ε_in))
    end
    return ψPlus
end

function new_EnergyState(ψPlusPrev_in, ψhPos_in)
    ψPlus_in = ψhPos_in
    if ψPlus_in >= ψPlusPrev_in
        ψPlus_out = ψPlus_in
    else
        ψPlus_out = ψPlusPrev_in
    end
    true, ψPlus_out
end

order = 2
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
reffe = ReferenceFE(lagrangian, Float64, order)
V = FESpace(model, reffe, conformity=:L2)
function project(f, V, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    l(v) = ∫(v * f) * dΩ
    op = AffineFEOperator(a, l, V, V)
    fh = solve(op)
    fh
end

#Phase Space
reffe_PF = ReferenceFE(lagrangian, Float64, order)
V0_PF = TestFESpace(model, reffe_PF; conformity=:H1)
U_PF = TrialFESpace(V0_PF)
sh = FEFunction(V0_PF, ones(num_free_dofs(V0_PF)))

#Displacement Space
reffe_Disp = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V0_Disp = TestFESpace(model, reffe_Disp;
    conformity=:H1,
    dirichlet_tags=["top", "bottom"],
    dirichlet_masks=[(true, true), (true, true)])
uh = zero(V0_Disp)

function stepPhaseField(sh_in, ψPlusPrev_in, ftol)
    res_PF(s, ϕ) = ∫(G_cbulk * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψPlusPrev_in * s * ϕ + (G_cbulk / ls) * s * ϕ) * dΩ - ∫((G_cbulk / ls) * ϕ) * dΩ
    jac_PF(s, ds, ϕ) = ∫(G_cbulk * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψPlusPrev_in * ds * ϕ + (G_cbulk / ls) * ds * ϕ) * dΩ
    op_PF = FEOperator(res_PF, jac_PF, U_PF, V0_PF)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), ftol=ftol, iterations=5)
    solver_PF = FESolver(nls)
    sh_out, = solve!(sh_in, solver_PF, op_PF)
    return sh_out, norm(Gridap.Algebra.residual(op_PF, sh_out), Inf)
end

function stepDisp(uh_in, sh_in, vApp, ftol)
    uApp1(x) = VectorValue(0.0, vApp)
    uApp2(x) = VectorValue(0.0, 0.0)
    uApp3(x) = VectorValue(0.0, -vApp)
    U_Disp = TrialFESpace(V0_Disp, [uApp1, uApp2])
    res_Disp(u, v) = ∫((ε(v) ⊙ (σ_mod ∘ (ε(u), ε(uh_in), sh_in)))) * dΩ
    jac_Disp(u, du, v) = ∫((ε(v) ⊙ (σ_mod ∘ (ε(du), ε(uh_in), sh_in)))) * dΩ
    op_Disp = FEOperator(res_Disp, jac_Disp, U_Disp, V0_Disp)
    nls_Disp = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), ftol=ftol, iterations=5)
    solver_Disp = FESolver(nls_Disp)
    uh_out, = solve!(uh_in, solver_Disp, op_Disp)
    return uh_out, norm(Gridap.Algebra.residual(op_Disp, uh_out), Inf)
end

labels = get_face_labeling(model)

using Gridap.Geometry
#labels = get_face_labeling(model)
#dimension = 2
#tags = get_face_tag(labels,dimension)
#const CrackLine = get_tag_from_name(labels,"Interface")

function Gc(tag)
    Gc_by_tag = Float64[]
    for i = 1:length(tag)
        if tag[i] == CrackLine
            push!(Gc_by_tag, G_c)
        else
            push!(Gc_by_tag, G_cbulk)
        end
    end
    return Gc_by_tag
end

#LoadTagId = get_tag_from_name(labels, "LoadWall")
Γ_Load = BoundaryTriangulation(model, tags="top")
dΓ_Load = Measure(Γ_Load, degree)
n_Γ_Load = get_normal_vector(Γ_Load)

#LoadTagIdLower = get_tag_from_name(labels, "LoadWallLower")
#Γ_LoadLower = BoundaryTriangulation(model, tags=LoadTagIdLower)
#dΓ_LoadLower = Measure(Γ_LoadLower, degree)
#n_Γ_LoadLower = get_normal_vector(Γ_LoadLower)


# Adaptive time stepping parameters
vApp = 12.5e-4 # incrementally updated applied displacement, and this is the initial value
const vAppMax = 2e-3 # total displacement to apply
const deltav_min = 1e-7#1e-7  # minimum increment (stop if deltav goes below this value)
const deltav_max = 1e-5 #1e-4 maximum increment (do not let deltav grow above this value)
deltav = deltav_max # current increment, which is adaptive
const growth_rate = 1.2 # allow deltav to grow if current increment solved with no cutbacks
const innerMax = 20 # number of cycles to find converged residuals for both coupled PDEs
const tol = 1e-6 #1e-3 for 600 #1e-6 #1e-5 #1e-4
count = 1

Load = Float64[]
LoadLower = Float64[]
Displacement = Float64[]
ALoad = Float64[]
CL_new = Float64[]
push!(CL_new, CL)
push!(Load, 0.0)
push!(LoadLower, 0.0)
push!(Displacement, 0.0)
push!(ALoad, 0.0)
ψPlusPrev = CellState(0.0, dΩ)
tick()
while vApp .< vAppMax
    @printf("Entering displacement step %i with applied displacement %.7e\n", count, float(vApp))

    cut_back = false # flag for whether current increment has needed to cut back
    early_quit = false # flag for whether to stop incrementing
    for inner = 1:innerMax

        ψhPlusPrev = project(ψPlusPrev, V, dΩ)
        @printf("Cycle = %i\n", inner)
        println("Solving phase field")
        global sh, ResNormPhaseField = stepPhaseField(sh, ψhPlusPrev, tol)
        @printf("Residual of phase field = %.6e\n", ResNormPhaseField)
        println("Solving displacement")
        global uh, ResNormDisp = stepDisp(uh, sh, vApp, tol)
        @printf("Residual of displacement = %.6e\n", ResNormDisp)

        ψhPos_in = ψPos ∘ (ε(uh))
        update_state!(new_EnergyState, ψPlusPrev, ψhPos_in)

        if ResNormPhaseField < tol && ResNormDisp < tol
            break # no need to cycle if both PDEs are already adequately converged
        end

        if inner == innerMax # if final cycle and we haven't broken out of the loop yet...
            println("Cutting back.\n")
            cut_back = true
            vApp = vApp .- deltav # remove increment
            global deltav = deltav / 2 # cut increment in half

            if deltav < deltav_min
                println("Failed to find convergence.\n")
                early_quit = true
            end
        end

    end

    if !cut_back # if this increment solved without any cut backs...
        Node_Force = sum(∫(n_Γ_Load ⋅ (σ_mod ∘ (ε(uh), ε(uh), sh))) * dΓ_Load) * vb
        yell = push!(Load, Node_Force[2])

        Node_ForceLower = sum(∫(n_Γ_LoadLower ⋅ (σ_mod ∘ (ε(uh), ε(uh), sh))) * dΓ_LoadLower) * vb
        push!(LoadLower, abs(Node_ForceLower[2]))

        push!(Displacement, vApp)
        if mod(count, 20) == 0
            writevtk(Ω, "Gc test$count.vtu", cellfields=
            ["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ_elas ∘ (ε(uh))])
        end
        if vApp >= 0 #1e-3
            if mod(count, 10) == 0
                dataf = DataFrame(Displacement=Displacement, Force=Load, ForceLower=LoadLower)
                CSV.write("Gc test$count.csv", dataf)
            end
        end

        # prepare next increment
        global deltav = deltav * growth_rate # ... increase the increment
        if deltav > deltav_max
            global deltav = deltav_max # ... but not above the max allowed increment
        end

        global count = count .+ 1
    end
    global vApp = vApp .+ deltav # apply the increment

    if early_quit
        break
    end
end

tock()

writevtk(Ω, "Gc test.vtu", cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ_elas ∘ (ε(uh))])

#Comparison Plot
plt.plot(Displacement * 1e3, Load, Displacement * 1e3, LoadLower)
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.legend(["Numerical Model Thicker Arm", "Numerical Model Thinner Arm"])
plt.title("NLS Opening Displacement vs Reaction Force, Gc test")
plt.grid()
plt.show()

df = DataFrame(Displacement=Displacement, Force=Load, ForceLower=LoadLower)
CSV.write("Gc test.csv", df)