using Kinetic, Solaris, OrdinaryDiffEq
using KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: throttle, Adam
using Solaris.Optim: LBFGS

cd(@__DIR__)
include("../../nn.jl")

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
    nt = 30,
    Kn = 1,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.nt)
γ = heat_capacity_ratio(set.K, 3)

f0 = 0.5 * (1 / π)^1.5 .*
    (exp.(-(vs3.u .- 0.99) .^ 2) .+ exp.(-(vs3.u .+ 0.99) .^ 2)) .*
    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)

prim0 =
    conserve_prim(moments_conserve(f0, vs3.u, vs3.v, vs3.w, vs3.weights), γ)
M0 = (maxwellian(vs3.u, vs3.v, vs3.w, prim0))

mu_ref = ref_vhs_vis(set.Kn, set.α, set.ω)
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
    set.α,
)

# Boltzmann
prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, chi])
data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

# BGK
prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

data_boltz_1D = zeros(vs.nu*2, axes(data_boltz, 4))
data_bgk_1D = zeros(vs.nu*2, axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[1:vs.nu, j], data_boltz_1D[vs.nu+1:end, j] =
        reduce_distribution(data_boltz[:, :, :, j], vs3.v, vs3.w, vs2.weights)
    data_bgk_1D[1:vs.nu, j], data_bgk_1D[vs.nu+1:end, j] =
        reduce_distribution(data_bgk[:, :, :, j], vs3.v, vs3.w, vs2.weights)
end
f0_1D = reduce_distribution(f0, vs3.v, vs3.w, vs2.weights)
M0_1D = reduce_distribution(M0, vs3.v, vs3.w, vs2.weights)

X = Array{Float32}(undef, vs.nu*2, size(data_boltz_1D, 2))
for i in axes(X, 2)
    X[:, i] .= data_boltz_1D[:, i]
end
M = Array{Float32}(undef, vs.nu*2, size(X, 2))
for i in axes(M, 2)
    M[:, i] .= [M0_1D[1]; M0_1D[2]]
end
X = vcat(X, ones(Float32, 1, size(X, 2)) .* τ0)
X1 = deepcopy(X)

rhs3D = zeros(Float32, vs3.nu, vs3.nv, vs3.nw, size(X, 2))
for i in axes(rhs3D, 4)
    df = @view rhs3D[:, :, :, i]
    boltzmann_ode!(df, data_boltz[:, :, :, i], [kn_bzm, set.nm, phi, psi, chi], 0.0)
end

Y = Array{Float32}(undef, vs.nu*2, set.nt)
for j in axes(rhs3D, 4)
    Y[1:vs.nu, j], Y[vs.nu+1:end, j] = reduce_distribution(rhs3D[:, :, :, j], vs3.v, vs3.w, vs2.weights)
end
Y1 = deepcopy(Y)

@load "prototype2.jld2" nn u X Y

#X2 = hcat(X[:, 1:2], X1)
#Y2 = hcat(Y[:, 1:2], Y1)
X2 = X1
Y2 = Y1

L = size(Y2, 2)

function loss(p, x, y)
    pred = nn(x, p, vs, 5/3, VDF{2,1}, Class{2})
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
res = sci_train(loss, res.u, Adam(1e-4); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500)
#res = sci_train(loss, res.u, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 100)

X, Y = X2, Y2
u = res.u
@save "reinforce2.jld2" u X Y

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ, VDF{2,1}, Class{2})
end

ube = ODEProblem(dfdt, X2[1:end-1, 1], tspan, (nn, u, vs, 5/3))
sol = solve(ube, Midpoint(); saveat = tsteps)

idx = 2
plot(vs.u, sol.u[idx][1:vs.nu])
scatter!(vs.u, X2[1:vs.nu, idx], alpha = 0.5)
