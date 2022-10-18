using Kinetic, Solaris, OrdinaryDiffEq, CairoMakie, NipponColors
using KitBase.JLD2
using Flux: elu, relu

dc = dict_color()
tc = plot_color()
cd(@__DIR__)

set = (
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
    maxTime = 3,
    tnum = 31,
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

cd(@__DIR__)
@load "specialize.jld2" u

nm = vs.nu
nν = vs.nu + 1

mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, 3; bias = false))
νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
nn = BGKNet(mn, νn)

(nn::BGKNet)(x, p, vs, γ, ::Type{VDF{1,1}}, ::Type{Class{2}}) = begin
    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = f_maxwellian(f, vs, γ)
    y = f .- M
    z = vcat(y, τ)

    nm = param_length(nn.Mnet)
    α = nn.Mnet(y, p[1:nm])
    S = collision_invariant(α, vs)

    #return (relu(M .* S) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end]))))
    return (relu(M) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end]))))
end

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ, VDF{1,1}, Class{2})
end

ube = ODEProblem(dfdt, f0, tspan, (nn, u, vs, 3))
sol = solve(ube, Midpoint(); saveat = tsteps)

idx = 6
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_nubgk[:, idx]; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, data_bgk[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_t1.pdf", fig)

idx = 11
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_nubgk[:, idx]; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, data_bgk[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_t2.pdf", fig)

idx = 21
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_nubgk[:, idx]; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, data_bgk[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_t3.pdf", fig)

"""
interpret E-net
"""

#--- collision ---#
function get_collision(idx)
    τ1 = KB.νshakhov_relaxation_time(τ, vs.u, prim0)

    Qs = @. (M0 - data_nubgk[:, idx]) / τ1
    Qb = @. (M0 - data_nubgk[:, idx]) / τ0
    Qn = nn([data_nubgk[:, idx]; τ0], u, vs, γ, VDF{1,1}, Class{2})[:]

    return Qs, Qb, Qn
end

Qs, Qb, Qn = get_collision(1)
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{Q}", title = "")
    lines!(vs.u, Qs; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, Qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, Qn; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_collision_t0.pdf", fig)

Qs, Qb, Qn = get_collision(6)
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{Q}", title = "")
    lines!(vs.u, Qs; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, Qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, Qn; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_collision_t1.pdf", fig)

Qs, Qb, Qn = get_collision(11)
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{Q}", title = "")
    lines!(vs.u, Qs; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, Qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, Qn; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_collision_t2.pdf", fig)

#--- frequency ---#
function get_nu(idx)
    Qs, Qb, Qn = get_collision(idx)
    return @. Qs / (M0 - data_nubgk[:, idx])
end

idx = 1
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\nu", title = "")
    lines!(vs.u, 1 ./ τ1; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, 1 ./ τ; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, get_nu(idx); color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_frequency_t0.pdf", fig)

idx = 6
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\nu", title = "")
    lines!(vs.u, 1 ./ τ1; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, 1 ./ τ; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, get_nu(idx); color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_frequency_t1.pdf", fig)

idx = 11
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\nu", title = "")
    lines!(vs.u, 1 ./ τ1; color = dc["ro"], label = "ν-BGK")
    lines!(vs.u, 1 ./ τ; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, get_nu(idx); color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("nubgk_frequency_t2.pdf", fig)
