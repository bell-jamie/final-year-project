function elas_fourth_order_const_tensor(E::Float64, ν::Float64, planar_state::String)
    if planar_state == "PlaneStrain"
        C1111 = E / (1 - ν * ν)
        C1122 = (E * ν) / (1 - ν * ν)
        C1112 = 0.0
        C2222 = E / (1 - ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    elseif planar_state == "PlaneStress"
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

function step_phase_field(phase, ψ_prev, f_tol, debug)
    res(s, ϕ) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_prev * s * ϕ +
        (gc_bulk / ls) * s * ϕ) * dΩ - ∫((gc_bulk / ls) * ϕ) * dΩ
    jac(s, ds, ϕ) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψ_prev * ds * ϕ +
        (gc_bulk / ls) * ds * ϕ) * dΩ
    op = FEOperator(res, jac, phase.U, phase.V0)
    nls = NLSolver(show_trace=debug, method=:newton,
        linesearch=BackTracking(), ftol=f_tol, iterations=5)
    phase.sh, = solve!(phase.sh, FESolver(nls), op)
    return phase.sh, norm(residual(op, phase.sh), Inf)
end

function step_displacement(dΩ, disp, phase, v_app, f_tol, debug)
    disp.U = TrialFESpace(disp.V0, applyBCs(v_app))
    res(u, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    jac(u, du, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(du), ε(disp.uh), phase.sh))) * dΩ
    op = FEOperator(res, jac, disp.U, disp.V0)
    nls = NLSolver(show_trace = debug, method = :newton,
        linesearch = BackTracking(), ftol = f_tol, iterations = 5)
    disp.uh, = solve!(disp.uh, FESolver(nls), op_disp)
    return disp.uh, norm(residual(op, disp.uh), Inf)
end

function step_coupled_fields(dΩ, phase, disp, ψ_prev, v_app, f_tol, debug)
    disp.U = TrialFESpace(disp.V0, apply_BCs(v_app))
    V0 = MultiFieldFESpace([phase.V0, disp.V0])
    U = MultiFieldFESpace([phase.U, disp.U])
    res((s, u), (ϕ, v)) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(s) +
        2 * ψ_prev * s * ϕ + (gc_bulk / ls) * s * ϕ) * dΩ -
        ∫((gc_bulk / ls) * ϕ) * dΩ +
        ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    jac((s, u), (ds, du), (ϕ, v)) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(ds) +
        2 * ψ_prev * ds * ϕ + (gc_bulk / ls) * ds * ϕ) * dΩ +
        ∫(ε(v) ⊙ (σ_mod ∘ (ε(du), ε(disp.uh), phase.sh))) * dΩ
    op = FEOperator(res, jac, U, V0)
    nls = NLSolver(show_trace = debug, method = :newton,
        linesearch = BackTracking(), ftol = f_tol, iterations = NL_iters)
    sh_uh, = solve!(MultiFieldFEFunction([get_free_dof_values(phase.sh);
        get_free_dof_values(disp.uh)], V0, [phase.sh; disp.uh]), FESolver(nls), op)
    return sh_uh[1], sh_uh[2], norm(residual(op, sh_uh), Inf)
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

function σ(ε)
    C ⊙ ε
end

function construct_spaces(model::DiscreteModel)
    reffe = ReferenceFE(lagrangian, Float64, order)
    test = TestFESpace(model, reffe, conformity=:H1)
    trial = TrialFESpace(test)
    field = FEFunction(test, ones(num_free_dofs(test)))
    return PhaseFieldStruct(test, trial, field)
end

function construct_spaces(model::DiscreteModel, tags::Array{String}, masks::Array{Tuple{Bool, Bool}})
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    test = TestFESpace(model, reffe, conformity=:H1,
        dirichlet_tags=tags,
        dirichlet_masks=masks)
    trial = TrialFESpace(test)
    field = zero(test)
    return DispFieldStruct(test, trial, field)
end

function fetch_timer()
    time_current = peektimer()
    hours = floor(time_current / 3600)
    minutes = floor((time_current - hours * 3600) / 60)
    seconds = time_current - hours * 3600 - minutes * 60
    @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

function create_save_directory(filename::String)
    date = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")
    save_directory = joinpath(splitext(filename)[1] * "-files", date)

    if !isdir(save_directory)
        mkpath(save_directory)
    end
    save_directory
end

function NL_coupled_recursive()
    global count = 1
    global δv = δv_max
    global v_app = v_init
    global load = Float64[]; push!(load, 0.0)
    global displacement = Float64[]; push!(displacement, 0.0)
    global ψ_plus_prev = CellState(0.0, dΩ)
    # to revert to the previous state if the step fails
    global ψ_prev_step = CellState(0.0, dΩ)

    while v_app .< v_app_max
        if δv < δv_min
            @error "** δv < δv_min - solution failed **"
            @printf("\n------------------------------\n\n")
            break
        end

        global v_app += δv
        ψ_plus_prev = ψ_prev_step # remembers the last successful energy state
        @info "** Step: $count **" Time = fetchTimer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)

        for cycle ∈ 1:max_cycles
            global sh, pf_residual = stepPhaseField(sh, ψ_plus_prev, tol, false)
            global uh, disp_residual = stepDisplacement(uh, sh, v_app, tol, false)
            @info "Cycle: $cycle" S_Residual = @sprintf("%.3e", pf_residual) V_Residual = @sprintf("%.3e", disp_residual)

            # Update Energy State
            ψ_pos_in = ψPos ∘ ε(uh) # current energy state
            update_state!(newEnergyState, ψ_plus_prev, ψ_pos_in)

            # Check for convergence
            if pf_residual < tol && disp_residual < tol
                ψ_prev_step = ψ_plus_prev # saves energy state for the next step
                @info "** Step complete **"
                @printf("\n------------------------------\n\n")

                node_force = sum(∫(n_Γ_load ⋅ (σMod ∘ (ε(uh), ε(uh), sh))) * dΓ_load)
                push!(load, node_force[2])
                push!(displacement, v_app)

                if mod(count, 1) == 0 # write every nth iteration
                    writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                        cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])
                end

                data_frame = DataFrame(Displacement=displacement, Force=load)
                CSV.write(joinpath(save_directory, "loadDisplacement.csv"), data_frame)

                global δv = min(δv * growth_rate, δv_max)
                global count += 1
                break
            end

            # Check for cutback
            if cycle == max_cycles # adaptive cutback?
                @warn "** Max cycles - cutting back **"
                @printf("\n------------------------------\n\n")
                v_app -= δv # remove increment
                δv = δv / 2 # halve increment
            end
        end
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])
end

struct NLCoupledRecursiveProblem # problem struct for keep variables out of global scope
    E::Float64
    ν::Float64
    mesh::String
    BCTags::Array{String}
    BCMasks::Array{Tuple{Bool,Bool}}
    BCPositions::Array{Int}
end

function NL_coupled_multi_field()
    # Initialise domain, spaces, measures and boundary conditions
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    phase = construct_spaces(model)
    disp = construct_spaces(model, BCs.tags, BCs.masks)

    Γ_load = BoundaryTriangulation(model, tags = "load")
    dΓ_load = Measure(Γ_load, degree)
    n_Γ_load = get_normal_vector(Γ_load)

    # Initialise variables, arrays and cell states
    count = 1
    δv = δv_max
    v_app = v_init
    load = Float64[]; push!(load, 0.0)
    displacement = Float64[]; push!(displacement, 0.0)
    ψ_prev = CellState(0.0, dΩ)

    # Main loop
    while v_app .< v_app_max
        if δv < δv_min
            @error "** δv < δv_min - solution failed **"
            @printf("\n------------------------------\n\n")
            break
        end

        @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)
        global phase.sh, disp.uh, coupled_residual = step_coupled_fields(dΩ, phase, disp, ψ_prev, v_app, tol, true)

        if coupled_residual < tol
            @info "** Step complete **"
            @printf("\n------------------------------\n\n")

            # Update energy state - max(previous energy, current energy)
            update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

            node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load)
            push!(load, node_force[2]); push!(displacement, v_app)
            data_frame = DataFrame(Displacement = displacement, Force = load)
            CSV.write(joinpath(save_directory, "loadDisplacement.csv"), data_frame)

            if mod(count, 1) == 0 # write every nth iteration
                writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                    cellfields=["uh" => disp.uh, "s" => phase.sh,
                    "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
            end
            
            δv = min(δv * growth_rate, δv_max) # increase increment
            v_app += δv # add increment
            count += 1 # update count
        else
            @warn "** Step failed **"
            @printf("\n------------------------------\n\n")

            v_app -= δv # remove increment
            δv = δv / 2 # halve increment
        end
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => disp.uh, "s" => phase.sh,
        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
end

function plot_load_displacement(title::String)
    savefile = CSV.File(joinpath(save_directory, "loadDisplacement.csv"))
    displacement = savefile.Displacement; load = savefile.Force

    plt = plot(displacement * 1e3, load,
        xlabel="Displacement (mm)",
        ylabel="Load (N)",
        title=title,
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "loadDisplacementPlot.png"))
    display(plt)
end

mutable struct PhaseFieldStruct
    U::FESpace
    V0::FESpace
    sh::FEFunction
end

mutable struct DispFieldStruct
    U::FESpace
    V0::FESpace
    uh::FEFunction
end

mutable struct PFMProblem # this is left here as an idea - it seems that DrWatson.jl fills this purpose
    mesh::String
    E::Float64
    ν::Float64
    PlanarState::String
    η::Float64
    gc_bulk::Float64
    ls::Float64
    U_pf::FESpace
    V0_pf::FESpace
    U_disp::FESpace
    V0_disp::FESpace
end

struct boundary_conditions
    tags::Array{String}
    masks::Array{Tuple{Bool,Bool}}
    positions::Array{Int}
end