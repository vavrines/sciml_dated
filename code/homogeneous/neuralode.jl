# ------------------------------------------------------------
# Homogeneous relaxation: neural ode
# ------------------------------------------------------------

cd(@__DIR__)

using Revise
using Kinetic
using DifferentialEquations, Flux
using DiffEqFlux, Optim
using Plots, Dates
using FileIO, JLD2
using Printf

include("tools.jl")

# generate parameters
config = "config.txt"
D = read_dict(config)
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

tspan = (0.0, maxTime)
tRange = range(tspan[1], tspan[2], length=tlen)

γ = heat_capacity_ratio(inK, 3)
vSpace = VSpace3D(u0, u1, nu, v0, v1, nv, w0, w1, nw, vMeshType)

f0 = Float32.(0.5 * (1/π)^1.5 .* (exp.(-(vSpace.u .- 0.99).^2) .+ exp.(-(vSpace.u .+ 0.99).^2)) .*
     exp.(-vSpace.v.^2) .* exp.(-vSpace.w.^2)) |> Array
prim0 = conserve_prim(moments_conserve(f0, vSpace.u, vSpace.v, vSpace.w, vSpace.weights), γ)
M0 = Float32.(maxwellian(vSpace.u, vSpace.v, vSpace.w, prim0)) |> Array

mu_ref = ref_vhs_vis(knudsen, alpha, omega)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)
τ = mu_ref * 2. * prim0[end]^(0.5) / prim0[1];

phi, psi, phipsi = kernel_mode( nm, vSpace.u1, vSpace.v1, vSpace.w1, vSpace.du[1,1,1], vSpace.dv[1,1,1], vSpace.dw[1,1,1],
                                vSpace.nu, vSpace.nv, vSpace.nw, alpha );

function boltzmann!(df, f::Array{<:Real,3}, p, t)
    Kn, M, phi, psi, phipsi = p
    df .= boltzmann_fft(f, Kn, M, phi, psi, phipsi) # https://github.com/vavrines/Kinetic.jl
end

function bgk!(df, f::Array{<:Real,3}, p, t)
    g, tau = p
    df .= (g .- f) ./ tau
end

# Boltzmann
prob = ODEProblem(boltzmann!, f0, tspan, [kn_bzm, nm, phi, psi, phipsi])
data_boltz = solve(prob, Tsit5(), saveat=tRange) |> Array;

# BGK
prob1 = ODEProblem(bgk!, f0, tspan, [M0, τ])
data_bgk = solve(prob1, Tsit5(), saveat=tRange) |> Array;

data_boltz_1D = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(Float32, axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2), i in axes(data_boltz_1D, 1)
    data_boltz_1D[i,j] = sum(@. vSpace.weights[i,:,:] * data_boltz[i,:,:,j])
    data_bgk_1D[i,j] = sum(@. vSpace.weights[i,:,:] * data_bgk[i,:,:,j])
end

f0_1D = zeros(Float32, axes(f0, 1))
for i in axes(data_boltz_1D, 1)
    f0_1D[i] = sum(@. vSpace.weights[i,:,:] * f0[i,:,:])
end

M0_1D = zeros(Float32, axes(M0, 1))
for i in axes(M0, 1)
    M0_1D[i] = sum(@. vSpace.weights[i,:,:] * M0[i,:,:])
end

default(legend = true)
plot(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], data_boltz_1D[:,1], lw=2, label="Initial", color=:gray32)
plot!(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], data_boltz_1D[:,end], lw=2, label="t=8 Boltzmann")
plot!(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], data_bgk_1D[:,end], lw=2, label="t=8 BGK")
plot!(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], M0_1D[:], lw=2, label="Maxwellian", color=5)


#--- neural ode ---#
dudt = FastChain( (x, p) -> x.^2, # initial guess
                   FastDense(vSpace.nu, vSpace.nu*nh, tanh), # our 1-layer network
                   FastDense(vSpace.nu*nh, vSpace.nu) )
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=tRange)

function loss_n_ode(p)
    pred = n_ode(f0_1D, p)
    loss = sum(abs2, pred .- data_boltz_1D)
    #loss = sum(abs2, pred .- (data_boltz_1D .- data_bgk_1D))
    return loss, pred
end

cb = function (p, l, pred; doplot=true)
    display(l)
    # plot current prediction against dataset
    if doplot
        #pl = plot(tRange, data_boltz_1D[vSpace.nu÷2,:], lw=2, label="Exact")
        #scatter!(pl, tRange, pred[vSpace.nu÷2,:], lw=2, label="NN prediction")
        plt = contour(Array(tRange), vSpace.u[1:end,vSpace.nv÷2,vSpace.nw÷2], Array(pred), fill=false)
        display(plot(plt))
    end
    return false
end

res, anim = vis_train(loss_n_ode, n_ode.p, ADAM(), cb=cb, maxiters=1000)
gif(anim, "test.gif")

res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(), cb=cb, maxiters=200)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(), cb=cb, maxiters=500)
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb=cb, maxiters=500)

plt = plot(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], data_boltz_1D, label="FSM", color=:black, xlabel="X", ylabel="PDF")
#plt = plot!(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], data_bgk_1D, label="BGK", xlabel="X", ylabel="PDF")
plt = plot!(vSpace.u[:,vSpace.nv÷2,vSpace.nw÷2], n_ode(f0_1D, res.minimizer) |> Array, label="NBE")

savefig(plt, "pdf_series.pdf")

@save "nnpara,jld2" res

# benchmark
using BenchmarkTools
@benchmark n_ode(M0_1D, res.minimizer)
@benchmark solve(prob, Tsit5(), saveat=tRange) |> Array
@benchmark solve(prob1, Tsit5(), saveat=tRange) |> Array
