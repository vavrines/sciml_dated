using Kinetic, Solaris, OrdinaryDiffEq
using Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu, cpu, gpu, Data, throttle
using Solaris.Optim: LBFGS

cd(@__DIR__)
@load "prototype.jld2" nn u X Y

set = (
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
    maxTime = 10,
    tnum = 100,
    Kn = 1,
)

begin
    tspan = (0, set.maxTime)
    tsteps = linspace(tspan[1], tspan[2], set.tnum)
    γ = 3.0
    vs = VSpace1D(set.u0, set.u1, set.nu)
    momentquad = vs.u .* vs.weights
    energyquad = vs.u .^ 2 .* vs.weights

    f0 = @. 0.5 * (1 / π)^0.5 * (exp.(-(vs.u - 2) ^ 2) + 0.5 * exp(-(vs.u + 2) ^ 2))
    prim0 = conserve_prim(moments_conserve(f0, vs.u, vs.weights), γ)
    M0 = maxwellian(vs.u, prim0)

    mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]
    τ1 = KB.νshakhov_relaxation_time(ones(vs.nu) .* τ0, vs.u, prim0)

    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    prob2 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ1])
    data_νbgk = solve(prob2, Tsit5(), saveat = tsteps) |> Array
end

X1 = Array{Float64}(undef, vs.nu, size(data_νbgk, 2))
for i in axes(X1, 2)
    X1[:, i] .= data_νbgk[:, i]
end
τ = Array{Float64}(undef, 1, size(X1, 2))
for i in axes(τ, 2)
    τ[1, i] = τ0
end
X1 = vcat(X1, ones(Float64, 1, size(X1, 2)) .* τ[1])

Y1 = Array{Float64}(undef, vs.nu, size(data_νbgk, 2))
for i in axes(Y1, 2)
    df = @view Y1[:, i]
    f = data_νbgk[:, i]
    p = (M0, τ1)
    bgk_ode!(df, f, p, 0.0)
end

idx = rand() * 1000 |>  round |> Int
X2 = hcat(X[:, idx:idx+set.tnum÷10], X1)
Y2 = hcat(Y[:, idx:idx+set.tnum÷10], Y1)
#X2 = X1
#Y2 = Y1

cb = function (p, l)
    println("loss: $(l)")
    return false
end

L = size(Y2, 2)
function loss(p)
    pred = nn(X2, p, vs)
    r1 = sum(abs2, discrete_moments(pred, vs.weights)) / L
    r2 = sum(abs2, discrete_moments(pred, momentquad)) / L
    r3 = sum(abs2, discrete_moments(pred, energyquad)) / L

    return sum(abs2, pred - Y2) / L + (r1 + r2 + r3) * 1e-6
end

loss(u)

res = sci_train(loss, u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)
res = sci_train(loss, res.u, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 100)

X, Y = X2, Y2
u = res.u
@save "reinforce.jld2" u X Y
