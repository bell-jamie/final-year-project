function elasFourthOrderConstTensor(E::Float64, ν::Float64, PlanarState::String)
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

function volumetricDeviatoricProjection()
    I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
    I4 = I2 ⊗ I2
    I4_sym = one(SymFourthOrderTensorValue{2,Float64})
    I4_vol = (1.0 / 2) * I4
    I4_dev = I4_sym - I4_vol
    return I4_vol, I4_dev
end

#function σMod(ε::SymTensorValue{2, Float64, 3}, ε_in::SymTensorValue{2, Float64, 3}, s_in::Float64)
function σMod(ε, ε_in, s_in)
    if tr(ε_in) >= 0
        (s_in^2 + η) * σ(ε)
    else
        (s_in^2 + η) * I4_dev ⊙ σ(ε) + I4_vol ⊙ σ(ε)
    end
end

#function ψPos(ε_in::SymTensorValue{2, Float64, 3})
function ψPos(ε_in)
    if tr(ε_in) >= 0
        0.5 * (ε_in ⊙ σ(ε_in))
    else
        0.5 * (I4_dev ⊙ σ(ε_in) ⊙ (I4_dev ⊙ ε_in))
    end
end

#function newEnergyState(ψ_plus_prev_in::Float64, ψ_plus_in::CellState)
#=
function newEnergyState(ψ_plus_prev_in, ψ_plus_in)
    if ψ_plus_in >= ψ_plus_prev_in
        (true, ψ_plus_in)
    else
        (true, ψ_plus_prev_in)
    end
end
=#
function newEnergyState(ψ_plus_prev_in, ψ_plus_in)
    (true, max(ψ_plus_in, ψ_plus_prev_in))
end

function stepPhaseField(sh_in, ψ_plus_prev_in, f_tol, debug)
    res_pf(s, ϕ) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_plus_prev_in * s * ϕ + (gc_bulk / ls) * s * ϕ) * dΩ - ∫((gc_bulk / ls) * ϕ) * dΩ
    jac_pf(s, ds, ϕ) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψ_plus_prev_in * ds * ϕ + (gc_bulk / ls) * ds * ϕ) * dΩ
    op_pf = FEOperator(res_pf, jac_pf, U_pf, V0_pf)
    nls = NLSolver(show_trace=debug, method=:newton,
        linesearch=BackTracking(), ftol=f_tol, iterations=5)
    sh_out, = solve!(sh_in, FESolver(nls), op_pf)
    return sh_out, norm(residual(op_pf, sh_out), Inf)
end

function stepDisplacement(uh_in, sh_in, v_app, f_tol, debug)
    U_disp = TrialFESpace(V0_disp, applyBCs(v_app))
    res_disp(u, v) = ∫(ε(v) ⊙ (σMod ∘ (ε(u), ε(uh_in), sh_in))) * dΩ
    jac_disp(u, du, v) = ∫(ε(v) ⊙ (σMod ∘ (ε(du), ε(uh_in), sh_in))) * dΩ
    op_disp = FEOperator(res_disp, jac_disp, U_disp, V0_disp)
    nls_disp = NLSolver(show_trace=debug, method=:newton,
        linesearch=BackTracking(), ftol=f_tol, iterations=5)
    uh_out, = solve!(uh_in, FESolver(nls_disp), op_disp)
    return uh_out, norm(residual(op_disp, uh_out), Inf)
end

function stepCoupledFields(sh_in, uh_in, ψ_plus_prev_in, v_app, f_tol, debug)
    U_disp = TrialFESpace(V0_disp, applyBCs(v_app))
    V0_coupled = MultiFieldFESpace([V0_pf, V0_disp])
    U_coupled = MultiFieldFESpace([U_pf, U_disp])
    res_coupled((s, u), (ϕ, v)) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_plus_prev_in * s * ϕ + (gc_bulk / ls) * s * ϕ) * dΩ - ∫((gc_bulk / ls) * ϕ) * dΩ +
        ∫(ε(v) ⊙ (σMod ∘ (ε(u), ε(uh), sh))) * dΩ
    jac_coupled((s, u), (ds, du), (ϕ, v)) = ∫(gc_bulk * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψ_plus_prev_in * ds * ϕ + (gc_bulk / ls) * ds * ϕ) * dΩ +
        ∫(ε(v) ⊙ (σMod ∘ (ε(du), ε(uh), sh))) * dΩ
    op_coupled = FEOperator(res_coupled, jac_coupled, U_coupled, V0_coupled)
    nls_coupled = NLSolver(show_trace=debug, method=:newton,
        linesearch=BackTracking(), ftol=f_tol, iterations=20) #20 iterations!
    sh_uh_out, = solve!(MultiFieldFEFunction([get_free_dof_values(sh_in); get_free_dof_values(uh_in)], V0_coupled, [sh_in; uh_in]), FESolver(nls_coupled), op_coupled)
    return sh_uh_out, norm(residual(op_coupled, sh_uh_out), Inf)
end

function applyBCs(V_app)
    bc_bool = fill(false, 2 * length(bcs.tags))
    for position ∈ bcs.positions
        bc_bool[position] = true
    end
    conditions = []
    for pair ∈ eachslice(reshape(bc_bool, 2, :), dims = 2)
        if pair[1] && pair[2]
            push!(conditions, VectorValue(V_app, V_app))
        elseif pair[1] && !pair[2]
            push!(conditions, VectorValue(V_app, 0.0))
        elseif !pair[1] && pair[2]
            push!(conditions, VectorValue(0.0, V_app))
        else
            push!(conditions, VectorValue(0.0, 0.0))
        end
    end
    conditions
end

function σ(ε)
    C ⊙ ε
end

function constructPhaseFieldSpace()
    reffe_pf = ReferenceFE(lagrangian, Float64, order)
    V0_pf = TestFESpace(model, reffe_pf, conformity=:H1)
    U_pf = TrialFESpace(V0_pf)
    sh = FEFunction(V0_pf, ones(num_free_dofs(V0_pf)))
    return U_pf, V0_pf, sh
end

function constructDisplacementFieldSpace(tags::Array{String}, masks::Array{Tuple{Bool, Bool}})
    reffe_disp = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V0_disp = TestFESpace(model, reffe_disp, conformity=:H1,
        dirichlet_tags=tags,
        dirichlet_masks=masks)
    uh = zero(V0_disp)
    return V0_disp, uh
end

function fetchTimer()
    time_current = peektimer()
    hours = floor(time_current / 3600)
    minutes = floor((time_current - hours * 3600) / 60)
    seconds = time_current - hours * 3600 - minutes * 60
    @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

function createSaveDirectory(filename::String)
    date = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")
    save_directory = joinpath(splitext(filename)[1] * "-files", date)

    if !isdir(save_directory)
        mkpath(save_directory)
    end
    save_directory
end

function nonLinearRecursive()
    global count = 1
    global δv = δv_max
    global v_app = 0.0
    global load = Float64[]; push!(load, 0.0)
    global displacement = Float64[]; push!(displacement, 0.0)
    global ψ_plus_prev = CellState(0.0, dΩ)
    global ψ_prev_step = CellState(0.0, dΩ) # to revert to the previous state if the step fails

    while v_app .< v_app_max
        if δv < δv_min
            @error "** δv < δv_min - solution failed **"
            @printf("\n------------------------------\n\n")
            break
        end

        global v_app += δv
        ψ_plus_prev = ψ_prev_step # remembers the last successful energy state
        @info "** Step: $count **" Time=fetchTimer() Increment=@sprintf("%.3e mm", δv) Displacement=@sprintf("%.3e mm", v_app)
    
        for cycle ∈ 1:max_cycles
            global sh, pf_residual = stepPhaseField(sh, ψ_plus_prev, tol, false)
            global uh, disp_residual = stepDisplacement(uh, sh, v_app, tol, false)
            @info "Cycle: $cycle" S_Residual=@sprintf("%.3e", pf_residual) V_Residual=@sprintf("%.3e", disp_residual)
    
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

                if mod(count, 1) == 0 # write every 10th iteration
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

function nonLinearCoupled()
    global count = 1
    global δv = (δv_max + δv_min)/2
    global v_app = 0.0
    global load = Float64[]; push!(load, 0.0)
    global displacement = Float64[]; push!(displacement, 0.0)
    global ψ_plus_prev = CellState(0.0, dΩ)
    global ψ_prev_step = CellState(0.0, dΩ) # to revert to the previous state if the step fails

    while v_app .< v_app_max
        if δv < δv_min
            @error "** δv < δv_min - solution failed **"
            @printf("\n------------------------------\n\n")
            break
        end

        global v_app += δv
        ψ_plus_prev = ψ_prev_step # remembers the last successful energy state
        @info "** Step: $count **" Time=fetchTimer() Increment=@sprintf("%.3e mm", δv) Displacement=@sprintf("%.3e mm", v_app)

        # Solve Coupled PDEs
        sh_uh, coupled_residual = stepCoupledFields(sh, uh, ψ_plus_prev, v_app, tol, true)
        global sh = sh_uh[1]; global uh = sh_uh[2]

        # Update Energy State
        ψ_pos_in = ψPos ∘ ε(uh) # current energy state
        update_state!(newEnergyState, ψ_plus_prev, ψ_pos_in)

        @printf("\nCoupled Residual: %.3e\n", coupled_residual)

        if coupled_residual < tol
            ψ_prev_step = ψ_plus_prev # saves energy state for the next step
            @info "** Step complete **"
            @printf("\n------------------------------\n\n")

            node_force = sum(∫(n_Γ_load ⋅ (σMod ∘ (ε(uh), ε(uh), sh))) * dΓ_load)
            push!(load, node_force[2])
            push!(displacement, v_app)

            if mod(count, 1) == 0 # write every 10th iteration
                writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                    cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])
            end

            data_frame = DataFrame(Displacement=displacement, Force=load)
            CSV.write(joinpath(save_directory, "loadDisplacement.csv"), data_frame)

            global δv = min(δv * growth_rate, δv_max)
            global count += 1
        else
            @warn "** Step failed **"
            @printf("\n------------------------------\n\n")
            v_app -= δv # remove increment
            δv = δv / 2 # halve increment
        end
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => uh, "s" => sh, "epsi" => ε(uh), "sigma" => σ ∘ ε(uh)])
end

mutable struct pfm_problem # this is left here as an idea - it seems that DrWatson.jl fills this purpose
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
    masks::Array{Tuple{Bool, Bool}}
    positions::Array{Int}
end