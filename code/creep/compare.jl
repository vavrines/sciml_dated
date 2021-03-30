using Kinetic
using KitBase.ProgressMeter, KitBase.JLD2
using KitML.Flux, KitML.DiffEqFlux, KitBase.Plots
using BenchmarkTools

cd(@__DIR__)
begin
    set = Setup(
        "gas", # matter
        "creep", # case
        "2d2f2v", # space
        "kfvs", # flux
        "shakhov", # collision: for scalar conservation laws there are none
        1, # species
        2, # interpolation order
        "vanleer", # limiter
        "maxwell", # boundary
        0.8, # cfl
        100.0, # simulation time
    )

    ps = PSpace2D(0.0, 5.0, 200, 0.0, 1.0, 40)
    vs = VSpace2D(-5.0, 5.0, 28, -5.0, 5.0, 28)
    Kn = 0.064
    gas = KitBase.Gas(Kn, 0.0, 2/3, 1, 5/3, 0.81, 1.0, 0.5, ref_vhs_vis(Kn, 1.0, 0.5))

    prim0 = [1.0, 0.0, 0.0, 1.0]
    w0 = prim_conserve(prim0, 5/3)
    h0 = maxwellian(vs.u, vs.v, prim0)
    b0 = @. h0 * gas.K / (2 * prim0[end])
    bcL = deepcopy(prim0)
    bcR = [1.0, 0.0, 0.0, 273/573]
    ib = IB2F(w0, prim0, h0, b0, bcL, w0, prim0, h0, b0, bcR)

    ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)
    @load "continuum/ctr.jld2"
end

n0 = ks.vSpace.nu * ks.vSpace.nv * 2
nh = 4
nn = Chain(
    Dense(n0, n0*nh, tanh),
    #Dense(n0*nh, n0*nh, tanh),
    Dense(n0*nh, n0),
)

# use central point for benchmark
i = 100
j = 20
MH = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
MB = MH .* ks.gas.K ./ (2 * ctr[i, j].prim[end])
_tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
## shakhov
@benchmark begin
    _MH = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
    _MB = _MH .* ks.gas.K ./ (2 * ctr[i, j].prim[end])
    _tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
    _q = heat_flux(ctr[i, j].h, ctr[i, j].b, ctr[i, j].prim, ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights)
    _SH, _SB = shakhov(ks.vSpace.u, ks.vSpace.v, _MH, _MB, _q, ctr[i, j].prim, ks.gas.Pr, ks.gas.K)
    _Q = [(_SH - ctr[i, j].h)[:] ./ _tau; (_SB - ctr[i, j].b)[:] ./ _tau]
end
## UBE
@benchmark begin
    f0 = [MH[:]; MB[:]] .- [ctr[i, j].h[:]; ctr[i, j].b[:]]
    nn(f0) .+ f0 ./ _tau
end
## FSM
v3d = VSpace3D(ks.vSpace.u0, ks.vSpace.u1, ks.vSpace.nu, 
    ks.vSpace.v0, ks.vSpace.v1, ks.vSpace.nv, 
    ks.vSpace.v0, ks.vSpace.v1, ks.vSpace.nv, "rectangle")
phi, psi, phipsi = kernel_mode(8, v3d.u1, v3d.v1, v3d.w1, v3d.du[1,1,1], v3d.dv[1,1,1], v3d.dw[1,1,1],
    v3d.nu, v3d.nv, v3d.nw, 1.0)
@benchmark begin
    Ei = 0.5 * discrete_moments(ctr[i, j].b, ks.vSpace.u, ks.vSpace.weights, 0)
    λi = 0.5 * ctr[i, j].prim[1] / (ks.gas.γ - 1.0) / Ei / 3.0 * 2.0

    f0 = zeros(v3d.nu, v3d.nv, v3d.nw)
    for i3 in axes(f0, 3), i2 in axes(f0, 2), i1 in axes(f0, 1)
        f0[i1, i2, i3] = ctr[i, j].h[i1, i2] * (λi / π)^0.5 * exp(-λi * v3d.w[i1, i2, i3]^2)
    end
    boltzmann_fft(f0, ks.gas.Kn, 8, phi, psi, phipsi)
end
