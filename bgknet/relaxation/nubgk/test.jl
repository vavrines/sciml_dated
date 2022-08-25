using Kinetic, Solaris, OrdinaryDiffEq, CairoMakie
using Kinetic.KitBase.JLD2

set = (
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
    maxTime = 3,
    tnum = 16,
    Kn = 1,
)

begin
    tspan = (0, set.maxTime)
    tsteps = linspace(tspan[1], tspan[2], set.tnum)
    γ = 3.0
    vs = VSpace1D(set.u0, set.u1, set.nu)

    f0 = @. 0.5 * (1 / π)^0.5 * (exp.(-(vs.u - 2) ^ 2) + 0.5 * exp(-(vs.u + 2) ^ 2))
    prim0 = conserve_prim(moments_conserve(f0, vs.u, vs.weights), γ)
    M0 = maxwellian(vs.u, prim0)

    mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    q = heat_flux(f0, prim0, vs.u, vs.weights)
    S0 = shakhov(vs.u, M0, q, prim0, 2 / 3)

    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    prob2 = ODEProblem(bgk_ode!, f0, tspan, [M0 .+ S0, τ0])
    data_shakhov = solve(prob2, Tsit5(), saveat = tsteps) |> Array

    τ = ones(vs.nu) .* τ0
    τ1 = KB.νshakhov_relaxation_time(τ, vs.u, prim0)
    prob3 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ1])
    data_nubgk = solve(prob3, Tsit5(), saveat = tsteps) |> Array
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, τ; label = "constant")
    lines!(vs.u, τ1; label = "ν-BGK", linestyle = :dash)
    axislegend()
    fig
end

idx = 5
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_bgk[:, idx]; label = "f₀")
    #lines!(vs.u, data_shakhov[:, idx]; label = "Maxwellian", linestyle = :dash)
    lines!(vs.u, data_nubgk[:, idx]; label = "Shakhov", linestyle = :dashdot)
    axislegend()
    fig
end
