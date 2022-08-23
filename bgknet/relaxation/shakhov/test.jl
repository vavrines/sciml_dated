using Kinetic, Solaris, OrdinaryDiffEq, Plots
using Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam
using Solaris.Optim: LBFGS

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
end

plot(vs.u, f0)
plot(M0)
plot!(M0 + S0, line = :dash)

idx = 5
plot(data_bgk[:, idx])
plot!(data_shakhov[:, idx], line = :dash)

nm = vs.nu
nν = vs.nu + 1

mn = FnChain(FnDense(nm, nm*2, tanh; bias = false), FnDense(nm*2, nm*4, tanh; bias = false), FnDense(nm*4, nm; bias = false))
νn = FnChain(FnDense(nν, nν*2, tanh; bias = false), FnDense(nν*2, nν*4, tanh; bias = false), FnDense(nν*4, nm; bias = false))
#mn = FnChain(FnDense(nm, nm*2, tanh; bias = false), FnDense(nm*2, nm; bias = false))
#νn = FnChain(FnDense(nν, nν*2-1, tanh; bias = false), FnDense(nν*2-1, nm; bias = false))

nn = BGKNet(mn, νn)

cd(@__DIR__)
@load "minimizer.jld2" u
#@load "start.jld2" u

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ)
end

ube = ODEProblem(dfdt, f0, tspan, (nn, u, vs, 3))
sol = solve(ube, Midpoint(); saveat = tsteps)

plot(sol.u[10])
plot!(data_bgk[:, 10], line = :dash)
scatter!(data_shakhov[:, 10])
