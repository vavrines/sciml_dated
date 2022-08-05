# ============================================================
# BGKNet
# ReverseDiff as a workaround for mutating arrays
# ============================================================

using KitBase, Solaris, OrdinaryDiffEq, Plots
using ReverseDiff, Solaris.Optimization
using Solaris.Optim
using Solaris.Flux: sigmoid, Adam

cd(@__DIR__)
include("nn.jl")

const vs = VSpace1D(-6, 6, 40; precision = Float32)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

f0 = vs.u .^ 2 .* exp.(-vs.u .^ 2)
w0 = moments_conserve(f0, vs.u, vs.weights)
prim0 = conserve_prim(w0, 3)
M0 = maxwellian(vs.u, prim0)
q0 = heat_flux(f0, prim0, vs.u, vs.weights)
S0 = shakhov(vs.u, M0, q0, prim0, 2/3)
tspan = (0, 5)
tsteps = tspan[1]:0.1:tspan[2]

prob = ODEProblem(bgk_ode!, f0, tspan, (S0, 1.f0))
sol = solve(prob, Tsit5(); saveat = tsteps)

X = hcat(sol.u...)
Y = similar(X)
for j in axes(Y, 2)
    df = @view Y[:, j]
    bgk_ode!(df, X[:, j], (S0, 1.f0), 0.f0)
end

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu, sigmoid; bias = false))

nn = BGKNet(mn, νn)
p = init_params(nn)

data = (X, Y)
L = size(data[1], 2)
loss(p) = sum(abs2, nn(data[1], p, vs) - data[2]) / L

cb = function (p, l)
    println("loss: $(loss(p))")
    return false
end

res = sci_train(loss, p, Adam(), Optimization.AutoReverseDiff(); maxiters = 10000)
res = sci_train(loss, res.u, LBFGS(), Optimization.AutoReverseDiff(); maxiters = 500)

# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

y1 = nn(X[:, 1], res.u, vs)

plot(y1, lw = 2, label = "nn")
scatter!(Y[:, 1], label = "exact")
