using Kinetic, Solaris, OrdinaryDiffEq, Plots
using Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam
using Solaris.Optim: LBFGS

set = (
    maxTime = 3,
    tnum = 16,
    u0 = -8,
    u1 = 8,
    nu = 80,
    v0 = -8,
    v1 = 8,
    nv = 28,
    w0 = -8,
    w1 = 8,
    nw = 28,
    nm = 5,
    Kn = 1,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
    nh = 8,
)

begin
    tspan = (0, set.maxTime)
    tsteps = linspace(tspan[1], tspan[2], set.tnum) .|> Float32
    γ = heat_capacity_ratio(set.K, 3)
    vs = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

    f0 =
        Float32.(
            0.8 * (1 / π)^1.5 .*
            (exp.(-(vs.u .- 0.8) .^ 2) .+ exp.(-(vs.u .+ 0.8) .^ 2)) .*
            exp.(-vs.v .^ 2) .* exp.(-vs.w .^ 2),
        ) |> Array
    prim0 =
        conserve_prim(moments_conserve(f0, vs.u, vs.v, vs.w, vs.weights), γ)
    M0 = Float32.(maxwellian(vs.u, vs.v, vs.w, prim0)) |> Array

    mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
    kn_bzm = hs_boltz_kn(mu_ref, 1.0)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    phi, psi, phipsi = kernel_mode(
        set.nm,
        vs.u1,
        vs.v1,
        vs.w1,
        vs.du[1, 1, 1],
        vs.dv[1, 1, 1],
        vs.dw[1, 1, 1],
        vs.nu,
        vs.nv,
        vs.nw,
        set.alpha,
    )

    # Boltzmann
    prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, phipsi])
    data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

    # BGK
    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    data_boltz_1D = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
    data_bgk_1D = zeros(Float32, axes(data_bgk, 1), axes(data_bgk, 4))
    for j in axes(data_boltz_1D, 2)
        data_boltz_1D[:, j] .=
            reduce_distribution(data_boltz[:, :, :, j], vs.weights[1, :, :])
        data_bgk_1D[:, j] .=
            reduce_distribution(data_bgk[:, :, :, j], vs.weights[1, :, :])
    end
    f0_1D = reduce_distribution(f0, vs.weights[1, :, :])
    M0_1D = reduce_distribution(M0, vs.weights[1, :, :])
end

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu + 1, vs.nu * 2 + 1, tanh; bias = false), FnDense(vs.nu * 2 + 1, vs.nu, sigmoid; bias = false))
nn = BGKNet(mn, νn)

cd(@__DIR__)
@load "minimizer.jld2" u

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ)
end

vs1d = VSpace1D(set.u0, set.u1, set.nu; precision = Float32)
ube = ODEProblem(dfdt, f0_1D, tspan, (nn, u, vs1d, 3))
sol = solve(ube, Midpoint(); saveat = tsteps)

prob_bgk = ODEProblem(bgk_ode!, f0_1D, tspan, [M0_1D, τ0])
sol_bgk = solve(prob_bgk, Midpoint(); saveat = tsteps)

plot(sol.u[10])
plot!(sol_bgk.u[10], line = :dash)
scatter!(data_boltz_1D[:, 10])
