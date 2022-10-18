using Kinetic, Solaris
using KitBase.Distributions, KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu, Data, throttle
using Solaris.Optim: LBFGS

cd(@__DIR__)

set = config_ntuple(u0 = -8, u1 = 8, nu = 80)

vs = VSpace1D(set.u0, set.u1, set.nu)
momentquad = vs.u .* vs.weights
energyquad = vs.u .^ 2 .* vs.weights

@load "prototype.jld2" nn u X Y
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
    pred = nn(x, p, vs)
    #r1 = sum(abs2, discrete_moments(pred, vs.weights)) / L
    #r2 = 0#sum(abs2, discrete_moments(pred, momentquad)) / L
    #r3 = sum(abs2, discrete_moments(pred, energyquad)) / L

    #return sum(abs2, pred - y) / L + (r1 + r2 + r3) * 1e-6
    return sum(abs2, pred - y) / L
end
loss(p) = loss(p, X1, Y1)

his = Float64[]
his1 = Float64[]
cb = function (p, l)
    println("loss: $(l)")
    push!(his, l)

    y1 = loss(p, X2, Y2)
    push!(his1, y1)

    @save "optim_record.jld2" p his his1

    return false
end
cb = throttle(cb, 30)

global res = sci_train(
    loss,
    u,
    Adam(1e-4);
    cb = cb,
    ad = Optimization.AutoReverseDiff(),
    iters = 10000000,
)

@save "optim_record.jld2" res his his1
