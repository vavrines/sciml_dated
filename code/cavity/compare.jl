using Kinetic
using KitBase.ProgressMeter, KitBase.JLD2
using KitML.Flux, KitML.DiffEqFlux, KitBase.Plots
using BenchmarkTools

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")
@load "det.jld2" ctr

n0 = ks.vSpace.nu * ks.vSpace.nu
nh = 8
nn = Chain(
    Dense(n0, n0*nh, tanh),
    #FastDense(n0*nh, n0*nh, tanh),
    Dense(n0*nh, n0),
)

i = 23
j = 23
_m = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
_tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)

@benchmark begin
    nn((_m .- ctr[i, j].f)[:]) .+ (_m .- ctr[i, j].f)[:] ./ _tau
end