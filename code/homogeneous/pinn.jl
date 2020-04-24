# ------------------------------------------------------------
# Homogeneous relaxation: pinn
# ------------------------------------------------------------

using Revise
using Kinetic
using DifferentialEquations, Flux
using DiffEqFlux, NeuralNetDiffEq, Optim
using Plots, Dates
using FileIO, JLD2

#--- setup ---#
println("")
println("PINN solver\n")

config = "config.txt"
println("Reading settings from $(config):")

# generate parameters
D = read_dict(config)
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

tspan = (0.f0, Float32(maxTime))
dim = ifelse(parse(Int, space[3]) >= 3, 3, parse(Int, space[1]))
γ = 1.4#heat_capacity_ratio(inK, dim)
vSpace = VSpace1D(u0, u1, nu, vMeshType, nug)

f0 = Float32.(0.3 * vSpace.u .^ 2 .* exp.(-0.3 .* vSpace.u .^ 2)) |> Array
w0 = [
    discrete_moments(f0, vSpace.u, vSpace.weights, 0),
    discrete_moments(f0, vSpace.u, vSpace.weights, 1),
    discrete_moments(f0, vSpace.u, vSpace.weights, 2),
]
prim0 = conserve_prim(w0, γ)
M = Float32.(maxwellian(vSpace.u, prim0)) |> Array

bgk = (u, p, t) -> @. (p .- u)
prob = ODEProblem(ODEFunction(bgk), f0, tspan, M)
chain = Flux.Chain(Dense(1, 10, σ), Dense(10, nu))
opt = ADAM(0.1, (0.9, 0.95))
sol = solve(
    prob,
    NeuralNetDiffEq.NNODE(chain, opt, autodiff = true),
    verbose = true,
    dt = 0.5f0,
    maxiters = 1000,
)

plot(vSpace.u, sol[1], lw = 2, label = "initial")
plot!(vSpace.u, sol[10], lw = 2, label = "t=5")
plot!(vSpace.u, sol[end], lw = 2, label = "t=5")
