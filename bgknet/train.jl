using Kinetic, Solaris
using KitBase.Distributions, KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu, Data, throttle
using Solaris.Optim: LBFGS
using LinearAlgebra

cd(@__DIR__)

set = config_ntuple(u0 = -8, u1 = 8, nu = 80)

vs = VSpace1D(set.u0, set.u1, set.nu)
momentquad = vs.u .* vs.weights
energyquad = vs.u .^ 2 .* vs.weights

@load "../relaxation/shakhov/prototype.jld2" nn u X Y
up = deepcopy(u)

idx1 = sample(1:9999, 5000, replace = false)
idx2 = sample(1:9999, 1000, replace = false)

X1 = X[:, idx1]
Y1 = Y[:, idx1]
X2 = X[:, idx2]
Y2 = Y[:, idx2]

u = init_params(nn)
L = size(Y, 2)

function loss(p, x, y)
    δ = size(y, 2)
    pred = nn(x, p, vs)
    #r1 = sum(abs2, discrete_moments(pred, vs.weights)) / δ
    #r2 = sum(abs2, discrete_moments(pred, energyquad)) / δ

    r1 = norm(discrete_moments(pred, vs.weights)) / δ
    r2 = norm(discrete_moments(pred, energyquad)) / δ

    #return sum(abs2, pred - y) / δ + (r1 + r2) * 1e-6 + sum(abs2, p) * 1e-8
    return sum(abs2, pred - y) / size(y, 2)
    #return norm(pred - y) / δ + (r1 + r2) * 1e-6
end
#loss(p) = loss(p, X, Y)
loss(p) = loss(p, X1, Y1)

function loss1(p, x)
    δ = size(x, 2)
    pred = nn(x, p, vs)
    r1 = sum(abs2, discrete_moments(pred, vs.weights)) / δ
    r2 = sum(abs2, discrete_moments(pred, energyquad)) / δ
    return r1 + r2
end

function loss2(p)
    return norm(p) * 1e-6
end

his = Float64[]
his1 = Float64[]
his2 = Float64[]
his3 = Float64[]
cb = function (p, l)
    println("loss: $(l)")
    push!(his, l)

    y1 = loss(p, X2, Y2)
    push!(his1, y1)

    y2 = loss1(p, X1)
    push!(his2, y2)

    y3 = loss2(p)
    push!(his3, y3)

    @save "optim_record.jld2" p his his1 his2 his3

    return false
end
#cb = throttle(cb, 1)
cb = throttle(cb, 30)

global res = sci_train(
    loss,
    u,
    Adam(1e-4);
    cb = cb,
    ad = Optimization.AutoReverseDiff(),
    #ad = Optimization.AutoFiniteDiff(),
    iters = 10000000,
    #iters = 10,
)

#loss(u, X1, Y1)
