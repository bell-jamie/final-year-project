mutable struct PhaseField
    φ::FESpace
    s::FESpace
    sh::FEFunction
end

mutable struct DispField
    v::FESpace
    u::FESpace
    uh::FEFunction
end

struct BoundaryConditions
    tags::Array{String}
    masks::Array{Tuple{Bool,Bool}}
end

function stiffness_tensor(E::Float64, ν::Float64, planar_state::String)
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
    SymFourthOrderTensorValue(
        C1111, C1112, C1122,
        C1112, C1212, C2212,
        C1122, C2212, C2222)
end

function vol_dev()
    I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
    I4 = I2 ⊗ I2
    I4_sym = one(SymFourthOrderTensorValue{2,Float64})
    I4_vol = (1.0 / 2) * I4
    I4_dev = I4_sym - I4_vol
    return I4_vol, I4_dev
end

σ(ε) = (ℂ ⊙ ε)

ℌ(ψ⁰, ψ) = (true, max(ψ⁰, ψ))

σᵐᵒᵈ(ε, ε⁰, s) = tr(ε⁰) >= 0 ? (s^2 + η) * σ(ε) :
                 (s^2 + η) * (Iᵈᵉᵛ ⊙ σ(ε)) + Iᵛᵒˡ ⊙ σ(ε)

# Factor of two cancelled with 1/2 in weak form
ψ⁺(ε) = tr(ε) >= 0 ? ε ⊙ σ(ε) : (Iᵈᵉᵛ ⊙ σ(ε)) ⊙ (Iᵈᵉᵛ ⊙ ε)

function step_disp_field_lin!(dΩ, disp, phase, v_app)
    a(u, v) = ∫(ε(v) ⊙ (σᵐᵒᵈ ∘ (ε(u), ε(disp.uh), phase.sh))) * dΩ
    b(v) = 0.0
    disp.u = TrialFESpace(disp.v,
        [VectorValue(0.0, v_app), VectorValue(0.0, 0.0)])
    op = AffineFEOperator(a, b, disp.u, disp.v)
    disp.uh = solve(op)
    return norm(residual(op, disp.uh), Inf)
end

function step_phase_field_lin!(dΩ, phase, ψ)
    a(s, φ) = ∫(Gc * ls * ∇(φ) ⋅ ∇(s) + ψ * s * φ +
                (Gc / ls) * s * φ) * dΩ
    b(φ) = ∫((Gc / ls) * φ) * dΩ
    op = AffineFEOperator(a, b, phase.s, phase.φ)
    phase.sh = solve(op)
    return norm(residual(op, phase.sh), Inf)
end

function step_disp_field!(dΩ, disp, phase, v_app)
    σₘ(u) = σᵐᵒᵈ ∘ (ε(u), ε(disp.uh), phase.sh)
    res(u, v) = ∫(ε(v) ⊙ σₘ(u)) * dΩ
    jac(u, du, v) = ∫(ε(v) ⊙ σₘ(du)) * dΩ
    disp.u = TrialFESpace(disp.v,
        [VectorValue(0.0, v_app), VectorValue(0.0, 0.0)])
    op = FEOperator(res, jac, disp.u, disp.v)
    nls = NLSolver(show_trace=verbose, method=:newton,
        linesearch=BackTracking(), ftol=nl_tol, iterations=nl_iter)
    disp.uh, = solve!(disp.uh, FESolver(nls), op)
    return norm(residual(op, disp.uh), Inf)
end

function step_phase_field!(dΩ, phase, ψ)
    res(s, φ) = ∫(Gc * ls * ∇(φ) ⋅ ∇(s) + ψ * s * φ +
                  (Gc / ls) * s * φ) * dΩ - ∫((Gc / ls) * φ) * dΩ
    jac(s, ds, φ) = ∫(Gc * ls * ∇(φ) ⋅ ∇(ds) + ψ * ds * φ +
                      (Gc / ls) * ds * φ) * dΩ
    op = FEOperator(res, jac, phase.s, phase.φ)
    nls = NLSolver(show_trace=verbose, method=:newton,
        linesearch=BackTracking(), ftol=nl_tol, iterations=nl_iter)
    phase.sh, = solve!(phase.sh, FESolver(nls), op)
    return norm(residual(op, phase.sh), Inf)
end

function step_coupled_fields!(dΩ, disp, phase, ψ, v_app)
    σₘ(u, s) = σᵐᵒᵈ ∘ (ε(u), ε(disp.uh), s)
    ℌ(u) = ψ⁺ ∘ ε(u)
    res((u, s), (v, φ)) = ∫(ε(v) ⊙ σₘ(u, s)) * dΩ +
                          ∫(Gc * ls * ∇(φ) ⋅ ∇(s) + ℌ(u) * s * φ +
                            (Gc / ls) * s * φ) * dΩ - ∫((Gc / ls) * φ) * dΩ
    disp.u = TrialFESpace(disp.v,
        [VectorValue(0.0, v_app), VectorValue(0.0, 0.0)])
    test = MultiFieldFESpace([disp.v, phase.φ])
    trial = MultiFieldFESpace([disp.u, phase.s])
    op = FEOperator(res, trial, test)
    nls = NLSolver(show_trace=verbose, method=:newton,
        linesearch=BackTracking(), ftol=nl_tol, iterations=nl_iter)
    field = MultiFieldFEFunction([get_free_dof_values(disp.uh);
            get_free_dof_values(phase.sh)], test, [disp.uh, phase.sh])
    (disp.uh, phase.sh), = solve!(field, FESolver(nls), op)
    return norm(residual(op, field), Inf)
end

function construct_disp(model)
    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    test = TestFESpace(model, reffe, conformity=:H1,
        dirichlet_tags=bc.tags,
        dirichlet_masks=bc.masks)
    trial = TrialFESpace(test)
    field = FEFunction(test, zeros(num_free_dofs(test)))
    return DispField(test, trial, field) # v, u, uh
end

function construct_phase(model)
    reffe = ReferenceFE(lagrangian, Float64, order)
    test = TestFESpace(model, reffe, conformity=:H1)
    trial = TrialFESpace(test)
    field = FEFunction(test, ones(num_free_dofs(test)))
    return PhaseField(test, trial, field) # φ, s, sh
end

function linear_segregated()
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    phase = construct_phase(model)
    disp = construct_disp(model)
    ψ = CellState(0.0, dΩ)

    Γ_load = BoundaryTriangulation(model, tags="Load")
    dΓ_load = Measure(Γ_load, degree)
    n_Γ_load = get_normal_vector(Γ_load)

    count = 1
    δv = δv_coarse
    v_app = 0.0

    data_frame = DataFrame(Displacement=Float64[0.0], Increment=Float64[v_app],
        Force=Float64[0.0], Energy=Float64[0.0], Damage=Float64[1.0])

    # Main loop
    while v_app < v_app_max
        @info("** Step: $count **", t = fetch_timer(), δv = @sprintf("%.3e mm", δv),
            v = @sprintf("%.3e mm", v_app))

        for cycle ∈ 1:max_cycles
            err = abs(sum(∫(Gc * ls * ∇(phase.sh) ⋅ ∇(phase.sh) + 2 * ψ * phase.sh *
                                                                  phase.sh + (Gc / ls) * phase.sh * phase.sh) * dΩ -
                          ∫((Gc / ls) * phase.sh) * dΩ)) / abs(sum(∫((Gc / ls) * phase.sh) * dΩ))

            step_phase_field_lin!(dΩ, phase, ψ)
            step_disp_field_lin!(dΩ, disp, phase, v_app)
            update_state!(ℌ, ψ, ψ⁺ ∘ ε(disp.uh)) # Updates ψ to the maximum of both ψ & ψ⁺

            @info("Cycle: $cycle", ϵ = @sprintf("%.3e", err))

            if err < tol
                break
            end
        end

        @info "** Step complete **"
        @printf("\n------------------------------\n\n")

        load = sum(∫(n_Γ_load ⋅ (σᵐᵒᵈ ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) *
                   dΓ_load)[2]

        ψ_sum = sum(∫(ψ⁺ ∘ ε(disp.uh)) * dΩ)
        s_sum = sum(∫(phase.sh) * dΩ)
        push!(data_frame, (v_app, δv, load, ψ_sum, s_sum))
        write_data(Ω, disp.uh, phase.sh, data_frame, count, 10)

        δv = v_app < linear_region ? δv_coarse : δv_fine
        v_app += δv
        count += 1
    end

    write_data(Ω, disp.uh, phase.sh, data_frame, count, 1)
end

function non_linear_alternate_minimisation()
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    phase = construct_phase(model)
    disp = construct_disp(model)
    ψ = CellState(0.0, dΩ)

    Γ_load = BoundaryTriangulation(model, tags="Load")
    dΓ_load = Measure(Γ_load, degree)
    n_Γ_load = get_normal_vector(Γ_load)

    count = 1
    δv = δv_coarse
    v_app = 0.0

    data_frame = DataFrame(Displacement=Float64[0.0], Increment=Float64[v_app],
        Cycles=Int[0.0], Force=Float64[0.0], Energy=Float64[0.0],
        Damage=Float64[1.0])

    # Main loop
    while v_app < v_app_max
        @info("** Step: $count **", t = fetch_timer(), δv = @sprintf("%.3e mm", δv),
            v = @sprintf("%.3e mm", v_app))

        for cycle ∈ 1:max_cycles
            res_disp = step_disp_field!(dΩ, disp, phase, v_app)
            res_phase = step_phase_field!(dΩ, phase, ψ)
            update_state!(ℌ, ψ, ψ⁺ ∘ ε(disp.uh)) # Updates ψ to the maximum of both ψ & ψ⁺

            load = sum(∫(n_Γ_load ⋅ (σᵐᵒᵈ ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) *
                       dΓ_load)[2]

            @info("Cycle: $cycle", ϵs = @sprintf("%.3e", res_phase),
                ϵu = @sprintf("%.3e", res_disp), f = @sprintf("%3.1f N", load))

            # Check for convergence
            if res_phase < nl_tol && res_disp < nl_tol
                @info "** Step complete **"
                @printf("\n------------------------------\n\n")

                ψ_sum = sum(∫(ψ⁺ ∘ ε(disp.uh)) * dΩ)
                s_sum = sum(∫(phase.sh) * dΩ)
                push!(data_frame, (v_app, δv, cycle, load, ψ_sum, s_sum))
                write_data(Ω, disp.uh, phase.sh, data_frame, count, 10)

                δv = v_app < linear_region ? δv_coarse : δv_fine
                v_app += δv
                count += 1
                break  # Exit cycle
            end
        end
    end

    write_data(Ω, disp.uh, phase.sh, data_frame, count, 1)
end

function non_linear_monolithic()
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    phase = construct_phase(model)
    disp = construct_disp(model)
    ψ = CellState(0.0, dΩ)

    Γ_load = BoundaryTriangulation(model, tags="Load")
    dΓ_load = Measure(Γ_load, degree)
    n_Γ_load = get_normal_vector(Γ_load)

    count = 1
    δv = δv_coarse
    v_app = 0.0

    data_frame = DataFrame(Displacement=Float64[0.0], Increment=Float64[v_app],
        Cycles=Int[0.0], Force=Float64[0.0], Energy=Float64[0.0],
        Damage=Float64[1.0])

    # Main loop
    while v_app < v_app_max
        @info("** Step: $count **", t = fetch_timer(), δv = @sprintf("%.3e mm", δv),
            v = @sprintf("%.3e mm", v_app))

        res, = step_coupled_fields!(dΩ, disp, phase, ψ, v_app)
        update_state!(ℌ, ψ, ψ⁺ ∘ ε(disp.uh)) # Updates ψ to the maximum of both ψ & ψ⁺

        load = sum(∫(n_Γ_load ⋅ (σᵐᵒᵈ ∘ (ε(disp.uh), ε(disp.uh), phase.sh))) * dΓ_load)[2]
        ψ_sum = sum(∫(ψ⁺ ∘ ε(disp.uh)) * dΩ)
        s_sum = sum(∫(phase.sh) * dΩ)
        push!(data_frame, (v_app, δv, 0, load, ψ_sum, s_sum))
        write_data(Ω, disp.uh, phase.sh, data_frame, count, 10)

        δv = v_app < linear_region ? δv_coarse : δv_fine
        v_app += δv
        count += 1
    end

    write_data(Ω, disp.uh, phase.sh, data_frame, count, 1)
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
        joinpath(save_directory, basename(filename)))
    cp(joinpath("..", "pfm-lib.jl"), joinpath(save_directory, "pfm-lib.jl"))

    save_directory
end

function write_step_data(step::Int, load_frame::DataFrame)
    try
        CSV.write(joinpath(save_directory, "step-$step.csv"), load_frame)
    catch
        @error "Error writing to CSV file"
    end
end

function write_data(
    Ω::Gridap.Geometry.BodyFittedTriangulation, uh::FEFunction, sh::FEFunction,
    data_frame::DataFrame, count::Int, step::Int
)
    try
        CSV.write(joinpath(save_directory, "log.csv"), data_frame)
    catch
        @error "Error writing to CSV file"
    end

    if mod(count, step) == 0
        writevtk(Ω, joinpath(save_directory, "solution-iter-$count.vtu"),
            cellfields=["uh" => uh, "sh" => sh])
    end
end

function create_plots()
    savefile = CSV.File(joinpath(save_directory, "log.csv"))
    displacement = savefile.Displacement
    increment = savefile.Increment
    force = savefile.Force
    energy = savefile.Energy
    damage = savefile.Damage

    plt = plot(displacement * 1e3, force,
        xlabel="Displacement (mm)",
        ylabel="Force (N)",
        title="Force vs Displacement",
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "force-displacement.png"))

    plt = plot(displacement * 1e3, increment,
        xlabel="Displacement (mm)",
        ylabel="Increment",
        title="Increment vs Displacement",
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "increment-displacement.png"))

    plt = plot(displacement * 1e3, energy,
        xlabel="Displacement (mm)",
        ylabel="Energy",
        title="Energy vs Displacement",
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "energy-displacement.png"))

    plt = plot(displacement * 1e3, 1 .- damage,
        xlabel="Displacement (mm)",
        ylabel="Damage",
        title="Damage vs Displacement",
        legend=false,
        grid=true)

    savefig(plt, joinpath(save_directory, "damage-displacement.png"))
end
