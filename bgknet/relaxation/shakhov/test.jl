using Kinetic, Solaris, OrdinaryDiffEq, CairoMakie, NipponColors
using KitBase.JLD2
using Flux: elu, relu

dc = dict_color()
tc = plot_color()

set = config_ntuple(
    u0 = -8,
    u1 = 8,
    nu = 80,
    t1 = 3,
    nt = 31,
    Kn = 1,
)

begin
    tspan = (0, set.t1)
    tsteps = linspace(tspan[1], tspan[2], set.nt)
    γ = 3.0
    vs = VSpace1D(set.u0, set.u1, set.nu)

    f0 = @. 0.5 * (1 / π)^0.5 * (exp.(-(vs.u - 1.5) ^ 2) + 0.7 * exp(-(vs.u + 1.5) ^ 2))
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
@load "prototype.jld2" nn
@load "reinforce.jld2" u
@load "specialize.jld2" u

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ)
end

ube = ODEProblem(dfdt, f0, tspan, (nn, u, vs, 3))
sol = solve(ube, Midpoint(); saveat = tsteps)

idx = 6
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_shakhov[:, idx]; color = dc["ro"], label = "Shakhov")
    lines!(vs.u, data_bgk[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
#save("shakhov_t1.pdf", fig)

idx = 11
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_shakhov[:, idx]; color = dc["ro"], label = "Shakhov")
    lines!(vs.u, data_bgk[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
#save("shakhov_t2.pdf", fig)

idx = 21
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_shakhov[:, idx]; color = dc["ro"], label = "Shakhov")
    lines!(vs.u, data_bgk[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
#save("shakhov_t3.pdf", fig)

"""
interpret E-net
"""

data_shakhov



idx = 31
q = heat_flux(data_shakhov[:, idx], prim0, vs.u, vs.weights)
S = shakhov(vs.u, M0, q, prim0, 2 / 3)

Qs = @. (M0 + S - data_shakhov[:, idx]) / τ0
Qb = @. (M0 - data_shakhov[:, idx]) / τ0

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, Qs; color = dc["ro"], label = "Shakhov")
    lines!(vs.u, Qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    #scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end


Mn = nn_M(data_shakhov[:, idx])
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, M0; color = dc["ro"], label = "Shakhov")
    lines!(vs.u, M0 + S; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, Mn; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end

function nn_M(f)
    y = M0 - f
    relu(M0 .+ nn.Mnet(y, u[1:nm]))
end

function nn_tau(f)
    y = M0 - f
    z = vcat(y, τ0)
    (τ0 .* (1 .+ 0.9 .* elu.(nn.νnet(z, u[nm+1:end]))))
end

nn_tau(data_shakhov[:, idx])

nm = param_length(nn.Mnet)
