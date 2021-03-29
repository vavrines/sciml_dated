using Kinetic
using KitBase.ProgressMeter, KitBase.JLD2, KitBase.Plots
using KitML.Flux, KitML.DiffEqFlux
using BenchmarkTools

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")
@load "det.jld2" ctr

n0 = ks.vSpace.nu * ks.vSpace.nu
nh = 8
nn = FastChain(
    FastDense(n0, n0*nh, tanh),
    #FastDense(n0*nh, n0*nh, tanh),
    FastDense(n0*nh, n0),
)
@load "para_nn.jld2" u

i, j = 23, 23
_m = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
_tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
_q = heat_flux(ctr[i, j].f, ctr[i, j].prim, ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights)
_s = shakhov(ks.vSpace.u, ks.vSpace.v, _m, _q, ctr[i, j].prim, ks.gas.Pr)

_r0 = (_m - ctr[i, j].f) / _tau
_r1 = _s /_tau
_r2 = reshape(nn((_m .- ctr[i, j].f)[:], u), ks.vSpace.nu, :)

pic0 = contourf(_r0)
pic1 = contourf(_r0 + _r1)
pic2 = contourf(_r0 + _r2)

savefig(pic0, "cavity_bgk_center.pdf")
savefig(pic1, "cavity_shakhov_center.pdf")
savefig(pic2, "cavity_ube_center.pdf")