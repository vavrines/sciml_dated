using Kinetic, Solaris, OrdinaryDiffEq, CairoMakie
using KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: throttle, Adam
using Solaris.Optim: LBFGS

cd(@__DIR__)

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
    t1 = 10.0,
    tnum = 50,
    Kn = 1.0,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.tnum)
γ = heat_capacity_ratio(set.K, 3)

f0 = 0.5 * (1 / π)^1.5 .*
    (exp.(-(vs3.u .- 0.99) .^ 2) .+ exp.(-(vs3.u .+ 0.99) .^ 2)) .*
    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)

prim0 =
    conserve_prim(moments_conserve(f0, vs3.u, vs3.v, vs3.w, vs3.weights), γ)
M0 = (maxwellian(vs3.u, vs3.v, vs3.w, prim0))

mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)
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
    set.alpha,
)

# Boltzmann
prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, chi])
data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

# BGK
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
f0_1D = reduce_distribution(f0, vs2.weights)
M0_1D = reduce_distribution(M0, vs2.weights)

X = Array{Float32}(undef, vs.nu, size(data_boltz_1D, 2))
for i in axes(X, 2)
    X[:, i] .= data_boltz_1D[:, i]
end
M = Array{Float32}(undef, set.nu, size(X, 2))
for i in axes(M, 2)
    M[:, i] .= M0_1D
end
τ = Array{Float32}(undef, 1, size(X, 2))
for i in axes(τ, 2)
    τ[1, i] = τ0
end
X = vcat(X, ones(Float32, 1, size(X, 2)) .* τ[1])
X1 = deepcopy(X)

rhs3D = zeros(Float32, vs3.nu, vs3.nv, vs3.nw, size(X, 2))
for i in axes(rhs3D, 4)
    df = @view rhs3D[:, :, :, i]
    boltzmann_ode!(df, data_boltz[:, :, :, i], [kn_bzm, set.nm, phi, psi, chi], 0.0)
end

Y = Array{Float32}(undef, vs.nu, set.tnum)
for j in axes(rhs3D, 4)
    Y[:, j] .= reduce_distribution(rhs3D[:, :, :, j], vs2.weights)
end
Y1 = deepcopy(Y)

@load "prototype.jld2" nn u X Y

X2 = hcat(X[:, 1:2], X1)
Y2 = hcat(Y[:, 1:2], Y1)

L = size(Y2, 2)

function loss(p, x, y)
    pred = nn(x, p, vs)
    return sum(abs2, pred - y) / L
end
loss(p) = loss(p, X2, Y2)

loss(u)

cb = function (p, l)
    println("loss: $(l)")
    return false
end
cb = throttle(cb, 1)

res = sci_train(loss, u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500)
res = sci_train(loss, res.u, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500)

X, Y = X2, Y2
u = res.u
@save "reinforce.jld2" u X Y
