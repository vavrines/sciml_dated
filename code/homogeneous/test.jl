using Revise
using Kinetic
using DifferentialEquations, Flux
using DiffEqFlux, Optim
using Plots, Dates

function bgk!(du, u::Array{<:Real,1}, p, t)
    du .= p .- u
end

γ = 1.4
vSpace = VSpace1D(-7, 7, 41, "rectangle", 0)
nu = 41
nh = 300
tspan = (0.0, 8.0)
tRange = range(tspan[1], tspan[2], length=9)

f0 = Float32.(0.3 * vSpace.u .^ 2 .* exp.(-0.3 .* vSpace.u .^ 2)) |> Array
w0 = [
    discrete_moments(f0, vSpace.u, vSpace.weights, 0),
    discrete_moments(f0, vSpace.u, vSpace.weights, 1),
    discrete_moments(f0, vSpace.u, vSpace.weights, 2),
]
prim0 = conserve_prim(w0, γ)
M = Float32.(maxwellian(vSpace.u, prim0)) |> Array


prob = ODEProblem(bgk!, f0, tspan, M)
ode_data = solve(prob, Tsit5(), saveat=tRange) |> Array

dudt = FastChain( (x, p) -> x.^2,
                   FastDense(nu, nh, tanh),
                   FastDense(nh, nu) )
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=tRange)

function loss_n_ode(p)
    pred = n_ode(f0, p)
    loss = sum(abs2, pred .- ode_data)
    return loss, pred
end

cb = function (p, l, pred; doplot=true)
    display(l)
    # plot current prediction against dataset
    if doplot
        pl = plot(tRange, ode_data[41÷2+1,:], lw=2, label="Exact")
        scatter!(pl, tRange, pred[41÷2+1,:], lw=2, label="NN prediction")
        display(plot(pl))
    end
    return false
end

res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(), cb=cb, maxiters=200)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb=cb, maxiters=100)

plot(vSpace.u, n_ode(f0, res.minimizer).u)

plot(vSpace.u, n_ode(f0.*0.9, res.minimizer).u)
