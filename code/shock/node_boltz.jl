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

function boltzmann!(df, f, p, t)
    Kn, M, phi, psi, phipsi = p
    df .= boltzmann_fft(f, Kn, M, phi, psi, phipsi)
end

function bgk!(df, f, p, t)
    g, tau = p
    df .= (g .- f) ./ tau
end

D = read_dict("shock.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

γ = heat_capacity_ratio(inK, 3)
set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)
μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = GasProperty(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)
vSpace = VSpace3D(umin, umax, nu, vmin, vmax, nv, wmin, wmax, nw, vMeshType, nug, nvg, nwg)
wL, primL, fL, bcL, wR, primR, fR, bcR = ib_rh(mach, γ, vSpace.u, vSpace.v, vSpace.w)
ib = IB1D1F(wL, primL, fL, bcL, wR, primR, fR, bcR)

sos = sound_speed(ib.primR, γ)
vmax = vSpace.u1 + sos
tmax = vmax / pSpace.dx[1]
dt = set.cfl / tmax

tSpan = (0.f0, Float32(dt)*2)
tRan = range(tSpan[1], tSpan[2], length=tLen)
kn_bzm = hs_boltz_kn(gas.μᵣ, gas.Kn)

phi, psi, phipsi = kernel_mode(
	nm, vSpace.u1, vSpace.v1, vSpace.w1,
	vSpace.du[1,1,1], vSpace.dv[1,1,1], vSpace.dw[1,1,1],
	vSpace.nu, vSpace.nv, vSpace.nw, gas.αᵣ )

ML = Float32.(maxwellian(vSpace.u, vSpace.v, vSpace.w, ib.primL)) |> Array
MR = Float32.(maxwellian(vSpace.u, vSpace.v, vSpace.w, ib.primR)) |> Array

f1 = deepcopy(ML)
f2 = @. 0.8 * ML + 0.2 * MR
f3 = @. 0.5 * ML + 0.5 * MR
f4 = @. 0.2 * ML + 0.8 * MR
f5 = deepcopy(MR)

prim1 = deepcopy(ib.primL)
prim2 = conserve_prim(moments_conserve( f2, vSpace.u, vSpace.v, vSpace.w, vSpace.weights ), γ)
prim3 = conserve_prim(moments_conserve( f3, vSpace.u, vSpace.v, vSpace.w, vSpace.weights ), γ)
prim4 = conserve_prim(moments_conserve( f4, vSpace.u, vSpace.v, vSpace.w, vSpace.weights ), γ)
prim5 = deepcopy(ib.primR)

M1 = maxwellian(vSpace.u, vSpace.v, vSpace.w, prim1)
M2 = maxwellian(vSpace.u, vSpace.v, vSpace.w, prim2)
M3 = maxwellian(vSpace.u, vSpace.v, vSpace.w, prim3)
M4 = maxwellian(vSpace.u, vSpace.v, vSpace.w, prim4)
M5 = maxwellian(vSpace.u, vSpace.v, vSpace.w, prim5)

τ1 = vhs_collision_time(prim1, μᵣ, gas.ω)
τ2 = vhs_collision_time(prim2, μᵣ, gas.ω)
τ3 = vhs_collision_time(prim3, μᵣ, gas.ω)
τ4 = vhs_collision_time(prim4, μᵣ, gas.ω)
τ5 = vhs_collision_time(prim5, μᵣ, gas.ω)

prob = ODEProblem(boltzmann!, f1, tSpan, [kn_bzm, nm, phi, psi, phipsi])
#prob = ODEProblem(bgk!, f1, tSpan, [M1, τ1])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D1 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D1, 2)
    data_boltz_1D1[:,j] .= reduce_distribution(data_boltz[:,:,:,j], vSpace.weights, 1)
end

prob = ODEProblem(boltzmann!, f2, tSpan, [kn_bzm, nm, phi, psi, phipsi])
#prob = ODEProblem(bgk!, f2, tSpan, [M2, τ2])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D2 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D2, 2)
    data_boltz_1D2[:,j] .= reduce_distribution(data_boltz[:,:,:,j], vSpace.weights, 1)
end

prob = ODEProblem(boltzmann!, f3, tSpan, [kn_bzm, nm, phi, psi, phipsi])
#prob = ODEProblem(bgk!, f3, tSpan, [M3, τ3])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D3 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D3, 2)
    data_boltz_1D3[:,j] .= reduce_distribution(data_boltz[:,:,:,j], vSpace.weights, 1)
end

prob = ODEProblem(boltzmann!, f4, tSpan, [kn_bzm, nm, phi, psi, phipsi])
#prob = ODEProblem(bgk!, f4, tSpan, [M4, τ4])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D4 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D4, 2)
    data_boltz_1D4[:,j] .= reduce_distribution(data_boltz[:,:,:,j], vSpace.weights, 1)
end

prob = ODEProblem(boltzmann!, f5, tSpan, [kn_bzm, nm, phi, psi, phipsi])
#prob = ODEProblem(bgk!, f5, tSpan, [M5, τ5])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D5 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D5, 2)
    data_boltz_1D5[:,j] .= reduce_distribution(data_boltz[:,:,:,j], vSpace.weights, 1)
end

data_boltz_1D = [data_boltz_1D1, data_boltz_1D2, data_boltz_1D3, data_boltz_1D4, data_boltz_1D5]
f0_1D = [reduce_distribution(f1, vSpace.weights, 1),
		 reduce_distribution(f2, vSpace.weights, 1),
		 reduce_distribution(f3, vSpace.weights, 1),
		 reduce_distribution(f4, vSpace.weights, 1),
		 reduce_distribution(f5, vSpace.weights, 1)]
M0_1D = [reduce_distribution(M1, vSpace.weights, 1),
		 reduce_distribution(M2, vSpace.weights, 1),
		 reduce_distribution(M3, vSpace.weights, 1),
		 reduce_distribution(M4, vSpace.weights, 1),
		 reduce_distribution(M5, vSpace.weights, 1)]
ML_1D = reduce_distribution(ML, vSpace.weights, 1)
MR_1D = reduce_distribution(MR, vSpace.weights, 1)

#--- neural ode ---#
dudt = FastChain( (x, p) -> x.^3,
                   FastDense(vSpace.nu, vSpace.nu*12, tanh),
                   #FastDense(vSpace.nu*10, vSpace.nu*10, tanh),
                   FastDense(vSpace.nu*12, vSpace.nu) )
n_ode = NeuralODE(dudt, tSpan, Tsit5(), saveat=tRan)

function loss_n_ode(p)
    pred = [n_ode(f0_1D[1], p),
			n_ode(f0_1D[2], p),
			n_ode(f0_1D[3], p),
			n_ode(f0_1D[4], p),
			n_ode(f0_1D[5], p)]

    loss = sum(abs2, pred[1] .- data_boltz_1D[1]) +
		   sum(abs2, pred[2] .- data_boltz_1D[2]) +
		   sum(abs2, pred[3] .- data_boltz_1D[3]) +
		   sum(abs2, pred[4] .- data_boltz_1D[4]) +
		   sum(abs2, pred[5] .- data_boltz_1D[5])

    return loss, pred[1]
end

cb = function (p, l, pred; doplot=false)
    display(l)
    # plot current prediction against dataset
    if doplot
        pl = plot(tRan, data_boltz_1D[1][vSpace.nu÷2,:], lw=2, label="Exact")
        scatter!(pl, tRan, pred[vSpace.nu÷2,:], lw=2, label="NN")
        display(plot(pl))
    end
    return false
end

res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.002, (0.9, 0.95)), cb=cb, maxiters=200)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.001, (0.9, 0.999)), cb=cb, maxiters=300)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb=cb, maxiters=500)

plot(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], data_boltz_1D[3][:,:], lw=2, label="FSM")
plot(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], n_ode(f0_1D[3], res.minimizer).u[:], lw=2, label="NN")

@save "optimizer.jld2" res
