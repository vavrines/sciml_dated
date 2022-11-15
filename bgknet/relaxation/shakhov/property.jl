using Kinetic, OrdinaryDiffEq, CairoMakie, NipponColors
using KitBase.JLD2
using Flux: elu, relu

function shakhov_ode!(df, f, p, t)
    M, τ = p
    q = heat_flux(f, prim0, vs.u, vs.weights)
    S = shakhov(vs.u, M, q, prim0, 2 / 3)
    df .= (M .+ S .- f) ./ τ
end

dc = dict_color()
tc = plot_color()

set = config_ntuple(
    u0 = -8,
    u1 = 8,
    nu = 80,
    t1 = 5,
    nt = 51,
    Kn = 1,
)

begin
    tspan = (0, set.t1)
    tsteps = linspace(tspan[1], tspan[2], set.nt)
    γ = 3.0
    vs = VSpace1D(set.u0, set.u1, set.nu)

    f0 = @. 0.5 * (1 / π)^0.5 * (exp.(-(vs.u - 1.5) ^ 2) + 0.7 * exp(-(vs.u + 1.5) ^ 2))
    w0 = moments_conserve(f0, vs.u, vs.weights)
    prim0 = conserve_prim(w0, γ)
    M0 = maxwellian(vs.u, prim0)

    mu_ref = ref_vhs_vis(set.Kn, set.α, set.ω)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    q = heat_flux(f0, prim0, vs.u, vs.weights)
    S0 = shakhov(vs.u, M0, q, prim0, 2 / 3)

    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    prob2 = ODEProblem(shakhov_ode!, f0, tspan, [M0, τ0])
    data_shakhov = solve(prob2, Tsit5(), saveat = tsteps) |> Array
end

cd(@__DIR__)
@load "specialize_intrusive.jld2" u

nm = vs.nu
nν = vs.nu + 1

mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, 3; bias = false))
νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
nn = BGKNet(mn, νn)

(nn::BGKNet)(x, p, vs, γ, ::Type{VDF{1,1}}, ::Type{Class{3}}) = begin
    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = f_maxwellian(f, vs, γ)
    y = f .- M
    z = vcat(y, τ)

    nm = param_length(nn.Mnet)
    α = nn.Mnet(y, p[1:nm])
    S = collision_invariant(α, vs)

    #return (relu(M .* S) .- f) ./ τ
    return (relu(M0 .* S) .- f) ./ τ
end

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ, VDF{1,1}, Class{3})
end

ube = ODEProblem(dfdt, f0, tspan, (nn, u, vs, 3))
sol = solve(ube, Midpoint(); saveat = tsteps)

wseries = hcat([moments_conserve(sol.u[i], vs.u, vs.weights, VDF{1,1}) for i in eachindex(tsteps)]...)
wseries0 = hcat([moments_conserve(data_bgk[:, i], vs.u, vs.weights, VDF{1,1}) for i in eachindex(tsteps)]...)
wseries1 = hcat([moments_conserve(data_shakhov[:, i], vs.u, vs.weights, VDF{1,1}) for i in eachindex(tsteps)]...)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "t", ylabel = "Δρ", title = "")
    lines!(tsteps, wseries1[1, :]; color = dc["ro"], label = "Shakhov")
    lines!(tsteps, wseries0[1, :]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(tsteps, wseries[1, :]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("shakhov_density.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "t", ylabel = "ΔρE", title = "")
    lines!(tsteps, wseries1[3, :]; color = dc["ro"], label = "Shakhov")
    lines!(tsteps, wseries0[3, :]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(tsteps, wseries[3, :]; color = (dc["tokiwa"], 0.7), label = "UBE", linestyle = :dashdot)
    axislegend()
    fig
end
save("shakhov_energy.pdf", fig)
