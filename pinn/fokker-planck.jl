using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, Optim, GalacticOptim

@parameters x
@variables p(..)
@derivatives Dx'~x
@derivatives Dxx''~x

# PDE
α = 0.3
β = 0.5
_σ = 0.5

# Discretization
dx = 0.1

# here we use normalization condition: dx*p(x) ~ 1 in order to get a non-zero solution.
eq = [(α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x)), dx*p(x) ~ 1.]

# Initial and boundary conditions
bcs = [p(-2.2) ~ 0., p(2.2) ~ 0., p(-2.2) ~ p(2.2)]

# Space and time domains
domains = [x ∈ IntervalDomain(-2.2, 2.2)]

chain = FastChain(FastDense(1,12,Flux.σ), FastDense(12,12,Flux.σ), FastDense(12,1))

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy= NeuralPDE.GridTraining(dx=dx))

pde_system = PDESystem(eq,bcs,domains,[x],[p])
prob = NeuralPDE.discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=8000)
phi = discretization.phi

analytic_sol_func(x) = 28.022*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))

xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

using Plots
plot(xs ,u_real, label = "analytic")
plot!(xs ,u_predict, label = "predict")