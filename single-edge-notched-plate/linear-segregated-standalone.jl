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
using Dates

function elas_fourth_order_const_tensor(E::Float64, ν::Float64, planar_state::String)
    if planar_state == "PlaneStress"
        C1111 = E / (1 - ν * ν)
        C1122 = (E * ν) / (1 - ν * ν)
        C1112 = 0.0
        C2222 = E / (1 - ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    elseif planar_state == "PlaneStrain"
        C1111 = (E * (1 - ν * ν)) / ((1 + ν) * (1 - ν - 2 * ν * ν))
        C1122 = (E * ν) / (1 - ν - 2 * ν * ν)
        C1112 = 0.0
        C2222 = (E * (1 - ν)) / (1 - ν - 2 * ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    else
        error("Invalid planar state")
    end
    SymFourthOrderTensorValue(C1111, C1112, C1122,
        C1112, C1212, C2212,
        C1122, C2212, C2222)
end

function volumetric_deviatoric_projection()
    I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
    I4 = I2 ⊗ I2
    I4_sym = one(SymFourthOrderTensorValue{2,Float64})
    I4_vol = (1.0 / 2) * I4
    I4_dev = I4_sym - I4_vol
    return I4_vol, I4_dev
end

function σ(ε)
    C ⊙ ε
end

function σ_mod(ε, ε_in, s_in)
    if tr(ε_in) >= 0
        (s_in^2 + η) * σ(ε)
    else
        (s_in^2 + η) * I4_dev ⊙ σ(ε) + I4_vol ⊙ σ(ε)
    end
end

function ψ_pos(ε)
    if tr(ε) >= 0
        0.5 * (ε ⊙ σ(ε))
    else
        0.5 * (I4_dev ⊙ σ(ε) ⊙ (I4_dev ⊙ ε))
    end
end

function new_energy_state(ψ_prev, ψ_current)
    (true, max(ψ_prev, ψ_current))
end

function step_phase_field(dΩ, phase, ψh_prev)
    a(s, ϕ) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψh_prev * s * ϕ + (Gc / ls) * s * ϕ) * dΩ
    b(ϕ) = ∫((Gc / ls) * ϕ) * dΩ
    op = AffineFEOperator(a, b, phase.U, phase.V0)
    solve(op)
end

function step_disp_field(dΩ, disp, phase, v_app)
    a(u, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    b(v) = 0.0
    disp.U = TrialFESpace(disp.V0, apply_BCs(v_app))
    op = AffineFEOperator(a, b, disp.U, disp.V0)
    solve(op)
end

function apply_BCs(v_app)
    BC_bool = fill(false, 2 * length(BCs.tags))
    for position ∈ BCs.positions
        BC_bool[position] = true
    end
    conditions = []
    for pair ∈ eachslice(reshape(BC_bool, 2, :), dims = 2)
        if pair[1] && pair[2]
            push!(conditions, VectorValue(v_app, v_app))
        elseif pair[1] && !pair[2]
            push!(conditions, VectorValue(v_app, 0.0))
        elseif !pair[1] && pair[2]
            push!(conditions, VectorValue(0.0, v_app))
        else
            push!(conditions, VectorValue(0.0, 0.0))
        end
    end
    conditions
end

function fetch_timer()
    time_current = peektimer()
    hours = floor(time_current / 3600)
    minutes = floor((time_current - hours * 3600) / 60)
    seconds = time_current - hours * 3600 - minutes * 60
    @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

function create_save_directory(filename::String)
    date_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    save_directory = joinpath(dirname(filename),
        "files",
        splitext(basename(filename))[1] * "-jl",
        date_time)

    if !isdir(save_directory)
        mkpath(save_directory)
    end

    # Copy input file to save directory as .txt
    cp(filename,
        joinpath(save_directory,
        (splitext(basename(filename))[1] * ".txt")),
        force = true)

    save_directory
end

function write_solution(Ω, disp, phase, count)
    writevtk(Ω, joinpath(save_directory, "solution-iter-$count.vtu"),
        cellfields = ["uh" => disp.uh, "s" => phase.sh,
        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
end

mutable struct PhaseFieldStruct
    V0::FESpace
    U::FESpace
    sh::FEFunction
end

mutable struct DispFieldStruct
    V0::FESpace
    U::FESpace
    uh::FEFunction
end

struct BoundaryConditions
    tags::Array{String}
    masks::Array{Tuple{Bool,Bool}}
    positions::Array{Int}
end

function construct_phase(model)
    reffe = ReferenceFE(lagrangian, Float64, order)
    test = TestFESpace(model, reffe, conformity = :H1)
    trial = TrialFESpace(test)
    field = FEFunction(test, ones(num_free_dofs(test)))
    return PhaseFieldStruct(test, trial, field) # V0, U, sh
end

function construct_disp(model, tags, masks)
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    test = TestFESpace(model, reffe, conformity = :H1,
        dirichlet_tags = tags,
        dirichlet_masks = masks)
    trial = TrialFESpace(test)
    field = zero(test)
    return DispFieldStruct(test, trial, field) # V0, U, uh
end

function project(q, model, dΩ)
    # This is inefficient and could be run once at the start
    reffe = ReferenceFE(lagrangian, Float64, order)
    V = FESpace(model, reffe, conformity = :L2)
    a(u, v) = ∫(u * v) * dΩ
    l(v) = ∫(v * q) * dΩ
    op = AffineFEOperator(a, l, V, V)
    solve(op)
end

function linear_segregated()
    # Initialise domain, spaces, measures and boundary conditions
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    phase = construct_phase(model)
    disp = construct_disp(model, BCs.tags, BCs.masks)

    Γ_load = BoundaryTriangulation(model, tags = "load")
    dΓ_load = Measure(Γ_load, degree)
    n_Γ_load = get_normal_vector(Γ_load)

    # Initialise variables, arrays and cell states
    count = 1
    δv = δv_coarse
    v_app = δv
    ψ_prev = CellState(0.0, dΩ)

    sim_log = DataFrame(Displacement = Float64[0.0], Increment = Float64[v_app],
        Force = Float64[0.0], Energy = Float64[0.0], Damage = Float64[1.0])

    # Main loop
    while v_app < v_app_max
        if v_app > v_app_threshold
            δv = δv_refined
        end

        @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)

        for cycle ∈ 1:max_cycles
            ψh_prev = project(ψ_prev, model, dΩ) # re-added this just to make sure it's not the cause of issues

            # while you're in here mucking around with energy states, make sure you add this to the other solvers if necessary - ψ * h * _prev 

            err = abs(sum(∫(Gc * ls * ∇(phase.sh) ⋅ ∇(phase.sh) + 2 * ψh_prev * phase.sh *
                phase.sh + (Gc / ls) * phase.sh * phase.sh) * dΩ -
                ∫((Gc / ls) * phase.sh) * dΩ)) / abs(sum(∫((Gc / ls) * phase.sh) * dΩ))

            phase.sh = step_phase_field(dΩ, phase, ψh_prev)
            disp.uh = step_disp_field(dΩ, disp, phase, v_app)

            @info "Cycle: $cycle" Relative_Error = @sprintf("%.3e", err)

            update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

            if err < tol
                break
            end
        end
        
        @info "** Step complete **"
        @printf("\n------------------------------\n\n")

        # log = update_sim_log(data_frame, v_app, δv, ψ_sum, s_sum, node_force)

        ψ_sum = sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ)
        s_sum = sum(∫(phase.sh) * dΩ)
        node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load) # potentially make these into a function ^^^

        push!(sim_log, (v_app, δv, node_force[2], ψ_sum, s_sum))

        try
            CSV.write(joinpath(save_directory, "log.csv"), sim_log) # make this into function
        catch
            @error "Error writing to CSV file"
        end

        if mod(count, 10) == 0 # write every nth iteration
            write_solution(Ω, disp, phase, count)
        end

        v_app += δv
        count += 1
    end

    write_solution(Ω, disp, phase, count)
end

## Constants
const E = 210e3
const ν = 0.3
const C = elas_fourth_order_const_tensor(E, ν, "PlaneStrain")
const I4_vol, I4_dev = volumetric_deviatoric_projection()

const ls = 0.0075
const Gc = 2.7
const η = 1e-15

const max_cycles = 10
const tol = 1e-8
const δv_refined = 1e-5
const δv_coarse = 1e-4
const v_app_threshold = 5e-3
const v_app_max = 7e-3

## Model Setup
mesh_file = joinpath(@__DIR__, "meshes", "notchedPlateTriangular.msh")
save_directory = create_save_directory(@__FILE__)
BCs = BoundaryConditions(["load", "fixed"], [(false, true), (true, true)], [2])
const order = 1
const degree = 2 * order

## Run
tick()
linear_segregated()
tock()