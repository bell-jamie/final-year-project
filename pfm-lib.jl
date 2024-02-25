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

function step_phase_field(dΩ, phase, ψ_prev)
    a(s, ϕ) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_prev * s * ϕ + (Gc / ls) * s * ϕ) * dΩ
    b(ϕ) = ∫((Gc /ls) * ϕ) * dΩ
    op = AffineFEOperator(a, b, phase.U, phase.V0)
    solve(op)
end

function step_phase_field(dΩ, phase, ψ_prev, f_tol, debug)
    res(s, ϕ) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_prev * s * ϕ +
        (Gc / ls) * s * ϕ) * dΩ - ∫((Gc / ls) * ϕ) * dΩ
    jac(s, ds, ϕ) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(ds) + 2 * ψ_prev * ds * ϕ +
        (Gc / ls) * ds * ϕ) * dΩ
    op = FEOperator(res, jac, phase.U, phase.V0)
    nls = NLSolver(show_trace = debug, method = :newton,
        linesearch = BackTracking(), ftol = f_tol, iterations = 5)
    phase.sh, = solve!(phase.sh, FESolver(nls), op)
    return phase.sh, norm(residual(op, phase.sh), Inf)
end

function step_disp_field(dΩ, disp, phase, v_app)
    a(u, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    b(v) = 0.0
    disp.U = TrialFESpace(disp.V0, apply_BCs(v_app))
    op = AffineFEOperator(a, b, disp.U, disp.V0)
    solve(op)
end

function step_disp_field(dΩ, disp, phase, v_app, f_tol, debug)
    #phase.U = TrialFESpace(phase.V0, [0])   # dodgy stuff!
    res(u, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    jac(u, du, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(du), ε(disp.uh), phase.sh))) * dΩ
    disp.U = TrialFESpace(disp.V0, apply_BCs(v_app))
    op = FEOperator(res, jac, disp.U, disp.V0)
    nls = NLSolver(show_trace = debug, method = :newton,
        linesearch = BackTracking(), ftol = f_tol, iterations = 5)
    disp.uh, = solve!(disp.uh, FESolver(nls), op)
    return disp.uh, norm(residual(op, disp.uh), Inf)
end

function step_coupled_fields(dΩ, phase, disp, ψ_prev, v_app, f_tol, debug)
    disp.U = TrialFESpace(disp.V0, apply_BCs(v_app))
    V0 = MultiFieldFESpace([phase.V0, disp.V0])
    U = MultiFieldFESpace([phase.U, disp.U])
    res((s, u), (ϕ, v)) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(s) +
        2 * ψ_prev * s * ϕ + (Gc / ls) * s * ϕ) * dΩ -
        ∫((Gc / ls) * ϕ) * dΩ +
        ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    jac((s, u), (ds, du), (ϕ, v)) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(ds) +
        2 * ψ_prev * ds * ϕ + (Gc / ls) * ds * ϕ) * dΩ +
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

function increment(δv::Float64)
    min(δv_max, δv * growth_rate)
end

function increment(δv::Float64, v_app::Float64)
    if v_app > v_app_threshold
        min(δv_refined, δv * growth_rate)
    else
        increment(δv)
    end
end

function increment(δv::Float64, history::Array{Float64}, aggression::Float64)
    # this needs momentum as it can't see ahead and therefore should be cautious // or we just let cutbacks deal with it
    current = history[end]
    previous = history[end - 1]
    grad_norm = (abs(previous - current) / previous) / δv
    @printf("Grad norm: %.3e\n", grad_norm)
    min(increment(δv), max(δv_min, δv ^ (1 / (aggression * grad_norm))))
end

function σ(ε)
    C ⊙ ε
end

function construct_phase(model)
    reffe = ReferenceFE(lagrangian, Float64, order)
    test = TestFESpace(model, reffe, conformity=:H1) #,
        #dirichlet_tags = ["crack"],
        #dirichlet_masks = [true])
    trial = TrialFESpace(test)
    field = FEFunction(test, ones(num_free_dofs(test)))
    return PhaseFieldStruct(test, trial, field)
end

function construct_disp(model, tags, masks)
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    test = TestFESpace(model, reffe, conformity=:H1,
        dirichlet_tags = tags,
        dirichlet_masks = masks)
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

    # Copy input file to save directory as .txt
    cp(filename, joinpath(save_directory, (splitext(basename(filename))[1] * ".txt")), force = true) # force true to avoid parallel error

    save_directory
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

    data_frame = DataFrame(Displacement = Float64[0.0], Increment = Float64[v_app],
        Force = Float64[0.0], Energy = Float64[0.0], Damage = Float64[1.0])

    # Main loop
    while v_app < v_app_max
        if v_app > v_app_threshold
            δv = δv_refined
        end

        @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)

        for cycle ∈ 1:max_cycles
            phase.sh = step_phase_field(dΩ, phase, ψ_prev)
            disp.uh = step_disp_field(dΩ, disp, phase, v_app)

            err = abs(sum(∫(Gc * ls * ∇(phase.sh) ⋅ ∇(phase.sh) + 2 * ψ_prev * phase.sh *
                phase.sh + (Gc / ls) * phase.sh * phase.sh) * dΩ -
                ∫((Gc / ls) * phase.sh) * dΩ)) / abs(sum(∫((Gc / ls) * phase.sh) * dΩ))

            @info "Cycle: $cycle" Relative_Error = @sprintf("%.3e", err)

            update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

            if err < tol
                break
            end
        end
        
        @info "** Step complete **"
        @printf("\n------------------------------\n\n")

        ψ_sum = sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ)
        s_sum = sum(∫(phase.sh) * dΩ)
        node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load)

        push!(data_frame, (v_app, δv, node_force[2], ψ_sum, s_sum))

        try
            CSV.write(joinpath(save_directory, "log.csv"), data_frame)
        catch
            @error "Error writing to CSV file"
        end

        if mod(count, 10) == 0 # write every nth iteration
            writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                cellfields = ["uh" => disp.uh, "s" => phase.sh,
                "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
        end

        v_app += δv
        count += 1
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => disp.uh, "s" => phase.sh,
        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
end

function linear_segregated_parallel(ranks)
    options = "-ksp_type cg -pc_type gamg -ksp_monitor"
    GridapPETSc.with(args = split(options)) do
        # Initialise domain, spaces, measures and boundary conditions
        model = GmshDiscreteModel(ranks, mesh_file)
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

        data_frame = DataFrame(Displacement = Float64[0.0], Increment = Float64[v_app],
            Force = Float64[0.0], Energy = Float64[0.0], Damage = Float64[1.0])

        # Main loop
        while v_app < v_app_max
            if v_app > v_app_threshold
                δv = δv_refined
            end

            @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)

            for cycle ∈ 1:max_cycles
                ################# PHASE FIELD #################
                a_pf(s, ϕ) = ∫(Gc * ls * ∇(ϕ) ⋅ ∇(s) + 2 * ψ_prev * s * ϕ + (Gc / ls) * s * ϕ) * dΩ
                b_pf(ϕ) = ∫((Gc /ls) * ϕ) * dΩ
                op_pf = AffineFEOperator(a_pf, b_pf, phase.U, phase.V0)
                phase.sh = solve(PETScLinearSolver(), op_pf)
                ###############################################

                ############# DISPLACEMENT FIELD ##############
                a_disp(u, v) = ∫(ε(v) ⊙ (σ_mod ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
                b_disp(v) = 0.0
                disp.U = TrialFESpace(disp.V0, apply_BCs(v_app))
                op_disp = AffineFEOperator(a_disp, b_disp, disp.U, disp.V0)
                disp.uh = solve(PETScLinearSolver(), op_disp)
                ###############################################

                err = abs(sum(∫(Gc * ls * ∇(phase.sh) ⋅ ∇(phase.sh) + 2 * ψ_prev * phase.sh *
                    phase.sh + (Gc / ls) * phase.sh * phase.sh) * dΩ -
                    ∫((Gc / ls) * phase.sh) * dΩ)) / abs(sum(∫((Gc / ls) * phase.sh) * dΩ))

                @info "Cycle: $cycle" Relative_Error = @sprintf("%.3e", err)

                update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

                if err < tol
                    break
                end
            end
            
            @info "** Step complete **"
            @printf("\n------------------------------\n\n")

            ψ_sum = sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ)
            s_sum = sum(∫(phase.sh) * dΩ)
            node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load)

            push!(data_frame, (v_app, δv, node_force[2], ψ_sum, s_sum))

            try
                CSV.write(joinpath(save_directory, "log.csv"), data_frame)
            catch
                @error "Error writing to CSV file"
            end

            if mod(count, 10) == 0 # write every nth iteration
                writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                    cellfields = ["uh" => disp.uh, "s" => phase.sh,
                    "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
            end

            v_app += δv
            count += 1
        end

        writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
            cellfields=["uh" => disp.uh, "s" => phase.sh,
            "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
    end
end

function NL_coupled_recursive()
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
    δv = v_init
    v_app = v_init
    ψ_prev_stable = CellState(0.0, dΩ)

    data_frame = DataFrame(Displacement = Float64[0.0], Increment = Float64[v_app],
        Force = Float64[0.0], Energy = Float64[0.0], Damage = Float64[1.0])

    # Main loop
    while v_app < v_app_max
        @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)
        ψ_prev = ψ_prev_stable # reset energy state

        for cycle ∈ 1:max_cycles
            phase.sh, pf_residual = step_phase_field(dΩ, phase, ψ_prev, tol, false)
            disp.uh, disp_residual = step_disp_field(dΩ, disp, phase, v_app, tol, false)
            @info "Cycle: $cycle" S_Residual = @sprintf("%.3e", pf_residual) V_Residual = @sprintf("%.3e", disp_residual)

            # Update energy state - max(previous energy, current energy)
            update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

            # Check for convergence
            if pf_residual < tol && disp_residual < tol
                @info "** Step complete **"
                @printf("\n------------------------------\n\n")

                ψ_sum = sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ)
                s_sum = sum(∫(phase.sh) * dΩ)
                node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load)

                push!(data_frame, (v_app, δv, node_force[2], ψ_sum, s_sum))
                ψ_prev_stable = ψ_prev # saves energy state for the next step

                try
                    CSV.write(joinpath(save_directory, "log.csv"), data_frame)
                catch
                    @error "Error writing to CSV file"
                end

                if mod(count, 10) == 0 # write every nth iteration
                    writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                        cellfields = ["uh" => disp.uh, "s" => phase.sh,
                        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
                end

                δv = increment(δv, v_app)
                v_app += δv
                count += 1
                break
            end

            if cycle != max_cycles
                continue
            end

            δv = δv / 2 # halve increment
            v_app -= δv # remove half increment

            if δv < δv_min
                @error "** δv < δv_min - solution failed (ish)**"
                @printf("\n------------------------------\n\n")
                return
            else
                @warn "** Step failed - cutback **"
                @printf("\n------------------------------\n\n")
            end
        end
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => disp.uh, "s" => phase.sh,
        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
end

function NL_coupled_multi_field()
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
    δv = δv_max
    v_app = v_init
    load = Float64[];               push!(load, 0.0)
    displacement = Float64[];       push!(displacement, 0.0)
    increments = Float64[];         push!(increments, v_init)
    energy= Float64[];              push!(energy, sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ))
    damage = Float64[];             push!(damage, sum(∫(phase.sh) * dΩ))
    ψ_prev = CellState(0.0, dΩ)

    # Main loop
    while v_app < v_app_max
        if δv < δv_min
            @error "** δv < δv_min - solution failed **"
            @printf("\n------------------------------\n\n")
            break
        end

        @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)
        phase.sh, disp.uh, coupled_residual = step_coupled_fields(dΩ, phase, disp, ψ_prev, v_app, tol, true)

        if coupled_residual < tol
            @info "** Step complete **"
            @printf("\n------------------------------\n\n")

            update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

            ψ_sum = sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ)
            s_sum = sum(∫(phase.sh) * dΩ)
            node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load) # you can directly use sum()[1]

            push!(energy, ψ_sum)
            push!(damage, s_sum)
            push!(load, node_force[2])
            push!(displacement, v_app)
            push!(increments, δv)

            data_frame = DataFrame(Displacement = displacement, Increment = increments,
                Force = load, Energy = energy, Damage = damage)

            try
                CSV.write(joinpath(save_directory, "log.csv"), data_frame)
            catch
                @error "Error writing to CSV file"
            end

            if mod(count, 10) == 0 # write every nth iteration
                writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                    cellfields = ["uh" => disp.uh, "s" => phase.sh,
                    "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
            end
            
            δv = min(δv * growth_rate, δv_max) # increase increment
            v_app += δv # add increment
            count += 1 # update count
        else
            @warn "** Step failed **"
            @printf("\n------------------------------\n\n")
            
            δv = δv / 2 # halve increment
            v_app -= δv # remove half increment
        end
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => disp.uh, "s" => phase.sh,
        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
end

function NL_coupled_multi_field_MOD()
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
    δv = δv_max
    v_app = v_init
    load = Float64[];           push!(load, 0.0)
    displacement = Float64[];   push!(displacement, 0.0)
    increments = Float64[];     push!(increments, v_init)
    energy= Float64[];          push!(energy, sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ))
    damage = Float64[];         push!(damage, sum(∫(phase.sh) * dΩ))
    ψ_prev = CellState(0.0, dΩ)

    # Main loop
    while v_app < v_app_max
        if δv < δv_min
            δv = δv_min
            @error "** δv < δv_min - solution failed (ish)**"
            @printf("\n------------------------------\n\n")
            #break
        end

        if v_app > 5e-3
            δv = min(δv, 1e-5) # set the maximum increment to 1e-5 like Rahaman
        end

        @info "** Step: $count **" Time = fetch_timer() Increment = @sprintf("%.3e mm", δv) Displacement = @sprintf("%.3e mm", v_app)
        phase.sh, disp.uh, coupled_residual = step_coupled_fields(dΩ, phase, disp, ψ_prev, v_app, tol, true)

        if coupled_residual < tol
            @info "** Step complete **"
            @printf("\n------------------------------\n\n")

            # Update energy state - max(previous energy, current energy)
            update_state!(new_energy_state, ψ_prev, ψ_pos ∘ ε(disp.uh))

            ψ_sum = sum(∫(ψ_pos ∘ ε(disp.uh)) * dΩ);    push!(energy, ψ_sum)
            s_sum = sum(∫(phase.sh) * dΩ);              push!(damage, s_sum)
            push!(increments, δv)

            node_force = sum(∫(n_Γ_load ⋅ (σ_mod ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load)
            push!(load, node_force[2]); push!(displacement, v_app)

            data_frame = DataFrame(Displacement = displacement, Increment = increments,
                Force = load, Energy = energy, Damage = damage)

            try
                CSV.write(joinpath(save_directory, "log.csv"), data_frame)
            catch
                @error "Error writing to CSV file"
            end

            if mod(count, 10) == 0 # write every nth iteration
                writevtk(Ω, joinpath(save_directory, "partialSolve$count.vtu"),
                    cellfields = ["uh" => disp.uh, "s" => phase.sh,
                    "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
            end
            
            δv = min(δv * growth_rate, δv_max) # increase increment
            v_app += δv # add increment
            count += 1 # update count
        else
            @warn "** Step failed **"
            @printf("\n------------------------------\n\n")
            
            v_app -= δv # remove full increment
            δv = δv / 2 # halve increment
        end
    end

    writevtk(Ω, joinpath(save_directory, "fullSolve.vtu"),
        cellfields=["uh" => disp.uh, "s" => phase.sh,
        "epsi" => ε(disp.uh), "sigma" => σ ∘ ε(disp.uh)])
end

function plot_load_displacement(title::String)
    savefile = CSV.File(joinpath(save_directory, "log.csv"))
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

function plot_damage_displacement(title::String)
    savefile = CSV.File(joinpath(save_directory, "log.csv"))
    displacement = savefile.Displacement; damage = savefile.Damage

    plt = plot(displacement * 1e3, 1 .- damage,
        xlabel="Displacement (mm)",
        ylabel="Damage",
        title=title,
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "damageDisplacementPlot.png"))
    display(plt)
end

function plot_energy_displacement(title::String)
    savefile = CSV.File(joinpath(save_directory, "log.csv"))
    displacement = savefile.Displacement; energy = savefile.Energy

    plt = plot(displacement * 1e3, energy,
        xlabel="Displacement (mm)",
        ylabel="Energy",
        title=title,
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "energyDisplacementPlot.png"))
    display(plt)
end

function plot_increment_displacement(title::String)
    savefile = CSV.File(joinpath(save_directory, "log.csv"))
    displacement = savefile.Displacement; increment = savefile.Increment

    plt = plot(displacement * 1e3, increment * 1e3,
        xlabel="Displacement (mm)",
        ylabel="Increment (mm)",
        title=title,
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "incrementDisplacementPlot.png"))
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

struct BoundaryConditions
    tags::Array{String}
    masks::Array{Tuple{Bool,Bool}}
    positions::Array{Int}
end