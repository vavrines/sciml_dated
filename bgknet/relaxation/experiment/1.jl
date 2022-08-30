using Kinetic, Solaris, OrdinaryDiffEq
using Solaris.Flux: Adam, throttle
using Solaris.Optimization, ReverseDiff

set = (
    u0 = -5,
    u1 = 5,
    nu = 80,
    v0 = -5,
    v1 = 5,
    nv = 28,
    w0 = -5,
    w1 = 5,
    nw = 28,
    nm = 5,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
    t1 = 8.0,
    tnum = 30,
    Kn = 1.0,
)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.tnum)
γ = heat_capacity_ratio(set.K, 3)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

f0 = 0.5 * (1 / π)^1.5 .*
    (exp.(-(vs3.u .- 1) .^ 2) .+ 0.5 .* exp.(-(vs3.u .+ 1) .^ 2)) .*
    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)

mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)

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
    set.alpha,
)
prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, chi])
data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array
data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .=
        reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
end

h0, b0 = reduce_distribution(f0, vs3.v, vs3.w, vs2.weights)
prim0 = conserve_prim(moments_conserve(h0, b0, vs.u, vs.weights), γ)
H0 = maxwellian(vs.u, prim0)
B0 = energy_maxwellian(H0, prim0, 2)

τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]
q = heat_flux(h0, b0, prim0, vs.u, vs.weights)
SH, SB = shakhov(vs.u, H0, B0, q, prim0, 2 / 3, 2)

prob1 = ODEProblem(bgk_ode!, [h0; b0], tspan, ([H0; B0], τ0))
data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

prob2 = ODEProblem(bgk_ode!, [h0; b0], tspan, ([H0 + SH; B0 + SB], τ0))
data_shakhov = solve(prob2, Tsit5(), saveat = tsteps) |> Array

nm = vs.nu * 2
nν = vs.nu * 2 + 1

mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm; bias = false))
νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
nn = BGKNet(mn, νn)
#nn = BGKReduceNet(mn, νn)
u = init_params(nn)

X1 = Array{Float64}(undef, vs.nu*2, size(data_boltz)[end])
for i in axes(X1, 2)
    X1[1:vs.nu, i], X1[vs.nu+1:end, i] = reduce_distribution(data_boltz[:, :, :, i], vs3.v, vs3.w, vs2.weights)
end
τ = Array{Float64}(undef, 1, size(X1, 2))
for i in axes(τ, 2)
    τ[1, i] = τ0
end
X1 = vcat(X1, ones(Float64, 1, size(X1, 2)) .* τ[1])

rhs3D = zeros(Float32, vs3.nu, vs3.nv, vs3.nw, size(X1, 2))
for i in axes(rhs3D, 4)
    df = @view rhs3D[:, :, :, i]
    boltzmann_ode!(df, data_boltz[:, :, :, i], [kn_bzm, set.nm, phi, psi, chi], 0.0)
end

Y1 = Array{Float64}(undef, vs.nu*2, size(data_shakhov, 2))
for j in axes(Y1, 2)
    Y1[1:vs.nu, j], Y1[vs.nu+1:end, j] = reduce_distribution(rhs3D[:, :, :, j], vs3.v, vs3.w, vs2.weights)
end

L = size(Y1, 2)

function loss(p, x, y)
    pred = nn(x, p, vs, γ)
    return sum(abs2, pred - y) / L
end
loss(p) = loss(p, X1, Y1)

loss(u)

cb = function (p, l)
    println("loss: $(l)")
    return false
end
cb = throttle(cb, 1)

res = sci_train(loss, u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500)
res = sci_train(loss, res.u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 2000)

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ)
end

ube = ODEProblem(dfdt, [h0; b0], tspan, (nn, res.u, vs, 5/3))
sol = solve(ube, Midpoint(); saveat = tsteps)

using Plots

idx = 12
begin
    plot(vs.u, sol[idx][1:vs.nu], color = :gray32, label = "NN")
    plot!(vs.u, data_bgk[1:vs.nu, idx], color = 1, label = "BGK")
    plot!(vs.u, data_shakhov[1:vs.nu, idx], color = 2, line = :dash, label = "Shakhov")
    scatter!(vs.u, data_boltz_1D[1:vs.nu, idx], color = 3, label = "Boltzmann", alpha = 0.6)
end
