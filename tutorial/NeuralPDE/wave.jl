using Kinetic, NeuralPDE, ModelingToolkit, Plots
using Kinetic.KitML.Flux, Kinetic.KitML.DiffEqFlux, Kinetic.KitML.Solaris.GalacticOptim
using KitBase.Optim
using ModelingToolkit: Interval

@parameters t, x
@variables u(..)
∂t, ∂x = Differential(t), Differential(x)

#2D PDE
eq = ∂t(u(t, x)) + ∂x(u(t, x)) ~ 0
domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]


#bcs = [u(t, 0) ~ u(t, 1), u(t, 1) ~ u(t, 0), u(0, x) ~ sin(2π * x)]
#bcs = [u(t, 0) ~ sin(-2π * t), u(t, 1) ~ sin(2π * (x - t)), u(0, x) ~ sin(2π * x)]
bcs = [u(t, 0) ~ sin(-2π * t), u(t, 1) ~ u(t, 0), u(0, x) ~ sin(2π * x)]
#bcs = [u(0, x) ~ sin(2π * x), u(t, 0) ~ u(t, 1), u(t, 1) ~ u(t, 0)]

# Space and time domains

# Discretization
dx = 0.05

# Neural network
chain = FastChain(FastDense(2, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
discretization = PhysicsInformedNN(chain, GridTraining(dx); init_params = initθ)

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

cb = function (p, l)
    println("loss: $l")
    return false
end

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters = 1200)

phi = discretization.phi
xs = 0:dx:1

sol0 = [phi([0, x], res.u)[1] for x in xs]
sol1 = [phi([0.5, x], res.u)[1] for x in xs]
plot(xs, sol0, label="t=0")
plot!(xs, sol1, label="t=0.5")
