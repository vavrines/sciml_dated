# ------------------------------------------------------------
# Normal shock structure
# ------------------------------------------------------------

using Revise
using DifferentialEquations
using Flux
using DiffEqFlux
using Optim
using Plots
using Dates
using FileIO
using JLD2

#using Kinetic
include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
using .Kinetic

function bgk!(df, f, p, t)
    g, tau = p
    df .= (g .- f) ./ tau
end

D = read_dict("shock1D.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

γ = 3.
set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)
μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = Gas(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)
vSpace = VSpace1D(umin, umax, nu, vMeshType, nug)
wL, primL, fL, bcL, wR, primR, fR, bcR = ib_rh(mach, γ, vSpace.u)
ib = IB1D1F(wL, primL, fL, bcL, wR, primR, fR, bcR)

sos = sound_speed(ib.primR, γ)
vmax = vSpace.u1 + sos
tmax = vmax / pSpace.dx[1]
dt = set.cfl / tmax

tSpan = (0.f0, Float32(dt)*2)
tRan = range(tSpan[1], tSpan[2], length=tLen)

ML = Float32.(maxwellian(vSpace.u, ib.primL)) |> Array
MR = Float32.(maxwellian(vSpace.u, ib.primR)) |> Array

f1 = deepcopy(ML)
f2 = @. 0.8 * ML + 0.2 * MR
f3 = @. 0.5 * ML + 0.5 * MR
f4 = @. 0.2 * ML + 0.8 * MR
f5 = deepcopy(MR)

prim1 = deepcopy(ib.primL)
prim2 = conserve_prim(moments_conserve( f2, vSpace.u, vSpace.weights ), γ)
prim3 = conserve_prim(moments_conserve( f3, vSpace.u, vSpace.weights ), γ)
prim4 = conserve_prim(moments_conserve( f4, vSpace.u, vSpace.weights ), γ)
prim5 = deepcopy(ib.primR)

M1 = maxwellian(vSpace.u, prim1)
M2 = maxwellian(vSpace.u, prim2)
M3 = maxwellian(vSpace.u, prim3)
M4 = maxwellian(vSpace.u, prim4)
M5 = maxwellian(vSpace.u, prim5)

τ1 = vhs_collision_time(prim1, μᵣ, gas.ω)
τ2 = vhs_collision_time(prim2, μᵣ, gas.ω)
τ3 = vhs_collision_time(prim3, μᵣ, gas.ω)
τ4 = vhs_collision_time(prim4, μᵣ, gas.ω)
τ5 = vhs_collision_time(prim5, μᵣ, gas.ω)

prob = ODEProblem(bgk!, f1, tSpan, [M1, τ1])
data_boltz1 = solve(prob, Tsit5(), saveat=tRan) |> Array;

prob = ODEProblem(bgk!, f2, tSpan, [M2, τ2])
data_boltz2 = solve(prob, Tsit5(), saveat=tRan) |> Array;

prob = ODEProblem(bgk!, f3, tSpan, [M3, τ3])
data_boltz3 = solve(prob, Tsit5(), saveat=tRan) |> Array;

prob = ODEProblem(bgk!, f4, tSpan, [M4, τ4])
data_boltz4 = solve(prob, Tsit5(), saveat=tRan) |> Array;

prob = ODEProblem(bgk!, f5, tSpan, [M5, τ5])
data_boltz5 = solve(prob, Tsit5(), saveat=tRan) |> Array;

data_boltz = [data_boltz1, data_boltz2, data_boltz3, data_boltz4, data_boltz5]
f0 = [f1, f2, f3, f4, f5]
M0 = [M1, M2, M3, M4, M5]


#--- neural ode ---#
dudt = FastChain( (x, p) -> x.^3,
                   FastDense(vSpace.nu, vSpace.nu*12, tanh),
                   #FastDense(vSpace.nu*10, vSpace.nu*10, tanh),
                   FastDense(vSpace.nu*12, vSpace.nu) )
n_ode = NeuralODE(dudt, tSpan, Tsit5(), saveat=tRan)

function loss_n_ode(p)
    pred = [n_ode(f0[1], p),
			n_ode(f0[2], p),
			n_ode(f0[3], p),
			n_ode(f0[4], p),
			n_ode(f0[5], p)]

    loss = sum(abs2, pred[1] .- data_boltz[1]) +
		   sum(abs2, pred[2] .- data_boltz[2]) +
		   sum(abs2, pred[3] .- data_boltz[3]) +
		   sum(abs2, pred[4] .- data_boltz[4]) +
		   sum(abs2, pred[5] .- data_boltz[5])

    return loss, pred[1]
end

cb = function (p, l, pred; doplot=false)
    display(l)
    # plot current prediction against dataset
    if doplot
        pl = plot(tRan, data_boltz[1][vSpace.nu÷2,:], lw=2, label="Exact")
        scatter!(pl, tRan, pred[vSpace.nu÷2,:], lw=2, label="NN")
        display(plot(pl))
    end
    return false
end

res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.002, (0.9, 0.95)), cb=cb, maxiters=200)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.001, (0.9, 0.999)), cb=cb, maxiters=300)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb=cb, maxiters=500)

plot(vSpace.u[:], data_boltz[3][:,:], lw=2, label="BGK")
plot(vSpace.u[:], n_ode(f0[5], res.minimizer).u[:], lw=2, label="NN")

@save "optimizer.jld2" res
