using KitBase
using KitBase.JLD2
#using NPZ
#using PyCall

using CairoMakie

plotly = pyimport("plotly")
cd(@__DIR__)

@load "bgk.jld2" ks ctr
ks1 = deepcopy(ks)
ctr1 = deepcopy(ctr)
@load "shakhov.jld2" ks ctr

u = zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx)
v = zero(u)
x = zero(u)
y = zero(u)
for i in axes(u, 1), j in axes(u, 2), k in axes(u, 3)
    u[i, :, :] .= ks.vs.u[i, 1]
    v[:, j, :] .= ks.vs.v[1, j]
    x[:, :, k] .= ks.ps.x[k, 1]
    y[:, :, k] .= ks.ps.y[1, k]
end

fs = [zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx), zeros(ks.vs.nu, ks.vs.nv, ks.ps.ny)]
fb = [zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx), zeros(ks.vs.nu, ks.vs.nv, ks.ps.ny)]
Ms = [zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx), zeros(ks.vs.nu, ks.vs.nv, ks.ps.ny)]
Mb = [zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx), zeros(ks.vs.nu, ks.vs.nv, ks.ps.ny)]
τs = [zeros(ks.ps.nx), zeros(ks.ps.ny)]
τb = [zeros(ks.ps.nx), zeros(ks.ps.ny)]
Qb = [zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx), zeros(ks.vs.nu, ks.vs.nv, ks.ps.ny)]
Qs = [zeros(ks.vs.nu, ks.vs.nv, ks.ps.nx), zeros(ks.vs.nu, ks.vs.nv, ks.ps.ny)]

# x direction
for k = 1:ks.ps.nx
    fs[1][:, :, k] .= ctr[k, 23].h
    fb[1][:, :, k] .= ctr1[k, 23].h
    Mb[1][:, :, k] .= maxwellian(ks.vs.u, ks.vs.v, ctr1[k, 23].prim)

    mh = maxwellian(ks.vs.u, ks.vs.v, ctr[k, 23].prim)
    mb = energy_maxwellian(mh, ctr[k, 23].prim, ks.gas.K)
    q = heat_flux(
        ctr[k, 23].h,
        ctr[k, 23].b,
        ctr[k, 23].prim,
        ks.vs.u,
        ks.vs.v,
        ks.vs.weights,
    )
    s = shakhov(ks.vs.u, ks.vs.v, mh, mb, q, ctr[k, 23].prim, ks.gas.Pr, ks.gas.K)
    Ms[1][:, :, k] .= mh .+ s[1]

    τs[1][k] = vhs_collision_time(ctr[k, 23].prim, ks.gas.μᵣ, ks.gas.ω)
    τb[1][k] = vhs_collision_time(ctr1[k, 23].prim, ks1.gas.μᵣ, ks1.gas.ω)

    Qs[1][:, :, k] .= (Ms[1][:, :, k] - fs[1][:, :, k]) / τs[1][k]
    Qb[1][:, :, k] .= (Mb[1][:, :, k] - fb[1][:, :, k]) / τb[1][k]
end

# y direction
for k = 1:ks.ps.ny
    fs[2][:, :, k] .= ctr[23, k].h
    fb[2][:, :, k] .= ctr1[23, k].h
    Mb[2][:, :, k] .= maxwellian(ks.vs.u, ks.vs.v, ctr1[23, k].prim)

    mh = maxwellian(ks.vs.u, ks.vs.v, ctr[23, k].prim)
    mb = energy_maxwellian(mh, ctr[23, k].prim, ks.gas.K)
    q = heat_flux(
        ctr[23, k].h,
        ctr[23, k].b,
        ctr[23, k].prim,
        ks.vs.u,
        ks.vs.v,
        ks.vs.weights,
    )
    s = shakhov(ks.vs.u, ks.vs.v, mh, mb, q, ctr[23, k].prim, ks.gas.Pr, ks.gas.K)
    Ms[2][:, :, k] .= mh .+ s[1]

    τs[2][k] = vhs_collision_time(ctr[23, k].prim, ks.gas.μᵣ, ks.gas.ω)
    τb[2][k] = vhs_collision_time(ctr1[23, k].prim, ks1.gas.μᵣ, ks1.gas.ω)

    Qs[2][:, :, k] .= (Ms[2][:, :, k] - fs[2][:, :, k]) / τs[2][k]
    Qb[2][:, :, k] .= (Mb[2][:, :, k] - fb[2][:, :, k]) / τb[2][k]
end

npzwrite(
    "pdf.npz",
    Dict(
        "u" => u,
        "v" => v,
        "x" => x,
        "y" => y,
        "qsx" => Qs[1],
        "qsy" => Qs[2],
        "qbx" => Qb[1],
        "qby" => Qb[2],
    ),
)
