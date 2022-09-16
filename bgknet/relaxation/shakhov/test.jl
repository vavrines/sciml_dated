using Kinetic, Solaris, OrdinaryDiffEq, CairoMakie
using KitBase.JLD2

set = config_ntuple(
    u0 = -8,
    u1 = 8,
    nu = 80,
    t1 = 3,
    nt = 16,
    Kn = 1,
)

begin
    tspan = (0, set.t1)
    tsteps = linspace(tspan[1], tspan[2], set.nt)
    γ = 3.0
    vs = VSpace1D(set.u0, set.u1, set.nu)

    f0 = @. 0.5 * (1 / π)^0.5 * (exp.(-(vs.u - 2) ^ 2) + 0.5 * exp(-(vs.u + 2) ^ 2))
    prim0 = conserve_prim(moments_conserve(f0, vs.u, vs.weights), γ)
    M0 = maxwellian(vs.u, prim0)

    mu_ref = ref_vhs_vis(set.Kn, set.α, set.ω)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    q = heat_flux(f0, prim0, vs.u, vs.weights)
    S0 = shakhov(vs.u, M0, q, prim0, 2 / 3)

    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    prob2 = ODEProblem(bgk_ode!, f0, tspan, [M0 .+ S0, τ0])
    data_shakhov = solve(prob2, Tsit5(), saveat = tsteps) |> Array
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, f0; label = "f₀")
    lines!(vs.u, M0; label = "Maxwellian", linestyle = :dash)
    lines!(vs.u, M0 + S0; label = "Shakhov", linestyle = :dashdot)
    axislegend()
    fig
end

cd(@__DIR__)
@load "prototype.jld2" nn u
@load "reinforce.jld2" u

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ)
end

ube = ODEProblem(dfdt, f0, tspan, (nn, u, vs, 3))
sol = solve(ube, Midpoint(); saveat = tsteps)

idx = 5
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, sol.u[idx]; label = "nn")
    lines!(vs.u, data_bgk[:, idx]; label = "BGK", linestyle = :dash)
    scatter!(vs.u, data_shakhov[:, idx]; label = "Shakhov", linestyle = :dashdot)
    axislegend()
    fig
end

#save("reinforce.png", fig)
