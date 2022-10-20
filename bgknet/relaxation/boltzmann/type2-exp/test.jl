"""
Architecture II

H-theorem preserving RelaxNet
"""

using Kinetic, Solaris, OrdinaryDiffEq
using KitBase.JLD2
using Solaris.Flux: throttle, Adam, Data, elu, relu
using CairoMakie, NipponColors

dc = dict_color()
cd(@__DIR__)
include("../../../nn.jl")

set = config_ntuple(
    u0 = -5,
    u1 = 5,
    nu = 80,
    v0 = -5,
    v1 = 5,
    nv = 28,
    w0 = -5,
    w1 = 5,
    nw = 28,
    t1 = 8,
    nt = 81,
    Kn = 1,
)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.nt)
γ = 5 / 3

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

f0 = 0.5 * (1 / π)^1.5 .*
    (exp.(-(vs3.u .- 1) .^ 2) .+ 0.7 .* exp.(-(vs3.u .+ 1) .^ 2)) .*
    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)
#f0 = 0.5 * (1 / π)^1.5 .*
#    (exp.(-(vs3.u .- 0.99) .^ 2) .+ exp.(-(vs3.u .+ 0.99) .^ 2)) .*
#    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)

mu_ref = ref_vhs_vis(set.Kn, set.α, set.ω)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)

w0 = moments_conserve(f0, vs3.u, vs3.v, vs3.w, vs3.weights)
prim0 = conserve_prim(w0, γ)
M0 = (maxwellian(vs3.u, vs3.v, vs3.w, prim0))
τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

phi, psi, chi = kernel_mode(
    set.nm,
    vs3.u1,
    vs3.v1,
    vs3.w1,
    vs3.du[1, 1, 1],
    vs3.dv[1, 1, 1],
    vs3.dw[1, 1, 1],
    vs3.nu,
    vs3.nv,
    vs3.nw,
    set.α,
)
prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, chi])
data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .=
        reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
    data_bgk_1D[:, j] .=
        reduce_distribution(data_bgk[:, :, :, j], vs2.weights)
end
h0_1D, b0_1D = reduce_distribution(f0, vs3.v, vs3.w, vs2.weights)
H0_1D, B0_1D = reduce_distribution(M0, vs3.v, vs3.w, vs2.weights)

@load "prototype2.jld2" nn u
@load "specialize.jld2" u

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ, VDF{2,1}, Class{2})
end

ube = ODEProblem(dfdt, [h0_1D; b0_1D], tspan, (nn, u, vs, 5/3))
sol = solve(ube, Midpoint(); saveat = tsteps)

idx = 2
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_boltz_1D[:, idx]; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, data_bgk_1D[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx][1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_t1.pdf", fig)

idx = 6
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_boltz_1D[:, idx]; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, data_bgk_1D[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx][1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_t2.pdf", fig)

idx = 11
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_boltz_1D[:, idx]; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, data_bgk_1D[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx][1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_t3.pdf", fig)

idx = 21
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_boltz_1D[:, idx]; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, data_bgk_1D[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx][1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_t4.pdf", fig)

idx = 41
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_boltz_1D[:, idx]; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, data_bgk_1D[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx][1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_t5.pdf", fig)

idx = 61
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, data_boltz_1D[:, idx]; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, data_bgk_1D[:, idx]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, sol.u[idx][1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_t6.pdf", fig)

"""
H-theorem
"""
h_boltz = zeros(set.nt)
h_bgk = zero(h_boltz)
h_nn = zero(h_boltz)
for i in eachindex(h_nn)
    h_boltz[i] = sum(@. data_boltz_1D[:, i] * log(data_boltz_1D[:, i]) * vs.weights)
    h_bgk[i] = sum(@. data_bgk_1D[:, i] * log(data_bgk_1D[:, i]) * vs.weights)
    h_nn[i] = sum(@. sol.u[i][1:vs.nu] * log(sol.u[i][1:vs.nu]) * vs.weights)
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "t", ylabel = "Entropy", title = "")
    lines!(tsteps, h_boltz; color = dc["ro"], label = "Boltzmann")
    lines!(tsteps, h_bgk; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(tsteps, h_nn; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend(; position=:lt)
    fig
end
save("boltz_entropy.pdf", fig)

"""
collision term
"""

idx = 1
begin
    _h, _b = reduce_distribution(data_boltz[:, :, :, idx], vs3.v, vs3.w, vs2.weights)
    q3 = zero(data_boltz[:, :, :, idx])
    boltzmann_ode!(q3, data_boltz[:, :, :, idx], [kn_bzm, set.nm, phi, psi, chi], 0.0)
    q1 = reduce_distribution(q3, vs2.weights)
    qb = (H0_1D - _h) ./ τ0
    qn = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{2})
end
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{Q}", title = "")
    lines!(vs.u, q1; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, qn[1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_collision_t1.pdf", fig)

idx = 11
begin
    _h, _b = reduce_distribution(data_boltz[:, :, :, idx], vs3.v, vs3.w, vs2.weights)
    q3 = zero(data_boltz[:, :, :, idx])
    boltzmann_ode!(q3, data_boltz[:, :, :, idx], [kn_bzm, set.nm, phi, psi, chi], 0.0)
    q1 = reduce_distribution(q3, vs2.weights)
    qb = (H0_1D - _h) ./ τ0
    qn = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{2})
end
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{Q}", title = "")
    lines!(vs.u, q1; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, qn[1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_collision_t2.pdf", fig)

idx = 21
begin
    _h, _b = reduce_distribution(data_boltz[:, :, :, idx], vs3.v, vs3.w, vs2.weights)
    q3 = zero(data_boltz[:, :, :, idx])
    boltzmann_ode!(q3, data_boltz[:, :, :, idx], [kn_bzm, set.nm, phi, psi, chi], 0.0)
    q1 = reduce_distribution(q3, vs2.weights)
    qb = (H0_1D - _h) ./ τ0
    qn = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{2})
end
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{Q}", title = "")
    lines!(vs.u, q1; color = dc["ro"], label = "Boltzmann")
    lines!(vs.u, qb; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(vs.u, qn[1:vs.nu]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("boltz_collision_t3.pdf", fig)

"""
open the black box
"""

(nn::BGKNet)(x, p, vs, γ, ::Type{VDF{2,1}}, ::Type{Class{3}}) = begin
    f = @view x[begin:end-1, :]
    h = @view x[begin:vs.nu, :]
    b = @view x[vs.nu+1:end-1, :]
    τ = @view x[end:end, :]

    H, B = f_maxwellian(h, b, vs, γ)
    M = [H; B]
    y = f .- M
    z = vcat(y, τ)

    nm = param_length(nn.Mnet)
    α = nn.Mnet(y, p[1:nm])
    SH = collision_invariant(α[1:3], vs)
    SB = collision_invariant(α[4:end], vs)
    S = vcat(SH, SB)

    return relu(M .* S), (τ .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end]))))
end

idx = 1
begin
    _h, _b = reduce_distribution(data_boltz[:, :, :, idx], vs3.v, vs3.w, vs2.weights)
    q3 = zero(data_boltz[:, :, :, idx])
    boltzmann_ode!(q3, data_boltz[:, :, :, idx], [kn_bzm, set.nm, phi, psi, chi], 0.0)
    q1 = reduce_distribution(q3, vs2.weights)
    qb = (H0_1D - _h) ./ τ0
    E, tau = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{3})
    qn = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{2})
    nu = qn[1:vs.nu] ./ (E[1:vs.nu] - _h)
    nu1 = q1 ./ (E[1:vs.nu] - _h)
end
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{E}")
    ax2 = Axis(fig[1, 1], ylabel = L"\nu", yaxisposition = :right, yticklabelcolor = dc["kohbai"])
    l1 = lines!(ax1, vs.u[18:63], E[18:63]; color = dc["ro"], label = "UBE")
    l2 = lines!(ax1, vs.u[18:63], H0_1D[18:63]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    l3 = lines!(ax2, vs.u[18:63], nu[18:63]; color = dc["kohbai"], label = "UBE")
    l4 = lines!(ax2, vs.u[18:63], ones(46) ./ τ0; color = dc["sohi"], label = "BGK", linestyle = :dash)
    axislegend(ax1; position = :lt)
    axislegend(ax2; position = :rt)
    fig
end
save("boltz_enu_t1.pdf", fig)

idx = 11
begin
    _h, _b = reduce_distribution(data_boltz[:, :, :, idx], vs3.v, vs3.w, vs2.weights)
    q3 = zero(data_boltz[:, :, :, idx])
    boltzmann_ode!(q3, data_boltz[:, :, :, idx], [kn_bzm, set.nm, phi, psi, chi], 0.0)
    q1 = reduce_distribution(q3, vs2.weights)
    qb = (H0_1D - _h) ./ τ0
    E, tau = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{3})
    qn = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{2})
    nu = qn[1:vs.nu] ./ (E[1:vs.nu] - _h)
    nu1 = q1 ./ (E[1:vs.nu] - _h)
end
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{E}")
    ax2 = Axis(fig[1, 1], ylabel = L"\nu", yaxisposition = :right, yticklabelcolor = dc["kohbai"])
    l1 = lines!(ax1, vs.u[18:63], E[18:63]; color = dc["ro"], label = "UBE")
    l2 = lines!(ax1, vs.u[18:63], H0_1D[18:63]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    l3 = lines!(ax2, vs.u[18:63], nu[18:63]; color = dc["kohbai"], label = "UBE")
    l4 = lines!(ax2, vs.u[18:63], ones(46) ./ τ0; color = dc["sohi"], label = "BGK", linestyle = :dash)
    axislegend(ax1; position = :lt)
    axislegend(ax2; position = :rt)
    fig
end
save("boltz_enu_t2.pdf", fig)

idx = 21
begin
    _h, _b = reduce_distribution(data_boltz[:, :, :, idx], vs3.v, vs3.w, vs2.weights)
    q3 = zero(data_boltz[:, :, :, idx])
    boltzmann_ode!(q3, data_boltz[:, :, :, idx], [kn_bzm, set.nm, phi, psi, chi], 0.0)
    q1 = reduce_distribution(q3, vs2.weights)
    qb = (H0_1D - _h) ./ τ0
    E, tau = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{3})
    qn = nn([_h; _b; τ0], u, vs, γ, VDF{2,1}, Class{2})
    nu = qn[1:vs.nu] ./ (E[1:vs.nu] - _h)
    nu1 = q1 ./ (E[1:vs.nu] - _h)
end
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "u", ylabel = L"\mathcal{E}")
    ax2 = Axis(fig[1, 1], ylabel = L"\nu", yaxisposition = :right, yticklabelcolor = dc["kohbai"])
    l1 = lines!(ax1, vs.u[18:63], E[18:63]; color = dc["ro"], label = "UBE")
    l2 = lines!(ax1, vs.u[18:63], H0_1D[18:63]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    l3 = lines!(ax2, vs.u[18:63], nu[18:63]; color = dc["kohbai"], label = "UBE")
    l4 = lines!(ax2, vs.u[18:63], ones(46) ./ τ0; color = dc["sohi"], label = "BGK", linestyle = :dash)
    axislegend(ax1; position = :lt)
    axislegend(ax2; position = :rt)
    fig
end
save("boltz_enu_t3.pdf", fig)
