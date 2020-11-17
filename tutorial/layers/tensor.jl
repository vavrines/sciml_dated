using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, LinearAlgebra

k, α, β, γ = 1, 0.1, 0.2, 0.3
tspan = (0.0, 10.0)

"""
x'' = −kx−αx³−βx'−γx'³

du[1] = x'
du[2] = x''
"""
function dxdt_train(du, u, p, t)
  du[1] = u[2]
  du[2] = -k*u[1] - α*u[1]^3 - β*u[2] - γ*u[2]^3
end

u0 = [1.0, 0.0]
ts = collect(0.0:0.1:tspan[2])
prob_train = ODEProblem{true}(dxdt_train, u0, tspan, p=nothing)
data_train = Array(solve(prob_train, Midpoint(), saveat=ts))

A = [LegendreBasis(10), LegendreBasis(10)]
nn = TensorLayer(A, 1)

f = x -> min(30one(x), x)

function dxdt_pred(du, u, p, t)
  du[1] = u[2]
  du[2] = -p[1]*u[1] - p[2]*u[2] + f(nn(u,p[3:end])[1])
end

#initial_params(nn)
α = zeros(102)

prob_pred = ODEProblem{true}(dxdt_pred, u0, tspan, p=nothing)

function predict_adjoint(θ)
    x = Array(solve(prob_pred,Tsit5(),p=θ,saveat=ts))
end

function loss_adjoint(θ)
    x = predict_adjoint(θ)
    loss = sum(norm.(x - data_train))
    return loss
end

function cb(θ, l)
    @show θ, l
    return false
end

res = DiffEqFlux.sciml_train(loss_adjoint, α, ADAM(0.05), cb = cb, maxiters = 150)
res = DiffEqFlux.sciml_train(loss_adjoint, res.minimizer, ADAM(0.001), cb = cb,maxiters = 300)

using Plots
data_pred = predict_adjoint(res.minimizer)
plot(ts, data_train[1,:], label = "X (ODE)")
plot!(ts, data_train[2,:], label = "V (ODE)")
plot!(ts, data_pred[1,:], label = "X (NN)")
plot!(ts, data_pred[2,:],label = "V (NN)")