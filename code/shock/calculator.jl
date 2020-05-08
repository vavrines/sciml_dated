using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0, 1.5f0)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

plot(t, ode_data[1,:])
plot!(t, ode_data[2,:])

dudt2 = FastChain((x,p) -> x.^3,
            FastDense(2,50,tanh),
            FastDense(50,2))
n_ode = NeuralODE(dudt2,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

cb = function (p,l,pred;doplot=false) #callback function to observe training
  display(l)
  # plot current prediction against data
  if doplot
    pl = scatter(t,ode_data[1,:],label="data")
    scatter!(pl,t,pred[1,:],label="prediction")
    display(plot(pl))
  end
  return false
end

res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.05), cb = cb, maxiters = 300)

res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb = cb)

plot(n_ode(u0, res.minimizer))




prob1 = ODEProblem(trueODEfunc, Float32[1.5; 0.5], tspan)
ode_data1 = solve(prob1, Tsit5(), saveat=t)
plot(ode_data1)

plot!(n_ode(Float32[1.5; 0.5], res.minimizer))


n_ode1 = NeuralODE(dudt2, (tspan[1], tspan[2]*2), Tsit5(), saveat=range(tspan[1], tspan[2]*2, length=datasize))
plot!(n_ode1(u0, res.minimizer))
