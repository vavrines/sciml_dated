# ------------------------------------------------------------
# Homogeneous relaxation: neural ode
# ------------------------------------------------------------

using Revise
using Kinetic
using DifferentialEquations, Flux
using DiffEqFlux, Optim
using Plots, Dates
using FileIO, JLD2

#--- setup ---#
function bgk!(du, u::Array{<:Real,1}, p, t)
    du .= p .- u
end

println("")
println("Neural ODE solver\n")

config = "config.txt"
println("Reading settings from $(config):")

# generate parameters
D = read_dict(config)
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

dim = ifelse(parse(Int, space[3]) >= 3, 3, parse(Int, space[1]))
γ = 1.4#heat_capacity_ratio(inK, dim)
vSpace = VSpace1D(u0, u1, nu, vMeshType, nug)

f0 = Float32.(0.3 * vSpace.u.^2 .* exp.(-0.3 .* vSpace.u.^2)) |> Array
w0 = [ discrete_moments(f0, vSpace.u, vSpace.weights, 0),
       discrete_moments(f0, vSpace.u, vSpace.weights, 1),
       discrete_moments(f0, vSpace.u, vSpace.weights, 2) ]
prim0 = conserve_prim(w0, γ)

M = Float32.(maxwellian(vSpace.u, prim0)) |> Array
f = similar(M)

#--- ode ---#
tspan = (0.0, maxTime)
tRange = range(tspan[1], tspan[2], length=tlen)
prob = ODEProblem(bgk!, f0, tspan, M)
ode_data = solve(prob, Tsit5(), saveat=tRange) |> Array

#--- neural ode ---#
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
        pl = plot(tRange, ode_data[nu÷2+1,:], lw=2, label="Exact")
        scatter!(pl, tRange, pred[nu÷2+1,:], lw=2, label="NN prediction")
        display(plot(pl))
    end
    return false
end

#--- training ---#
res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(), cb=cb, maxiters=mliter)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb=cb, maxiters=mliter÷2)

#--- output ---#
identifier = string(Dates.now(), "/")
outputFolder = replace(identifier, ":"=>".")
mkdir(outputFolder)
cp(config, string(outputFolder, "config.txt"))
fileOut = outputFolder * "data" * ".jld2"
@save fileOut vSpace, f0, tRange, ode_data, res
