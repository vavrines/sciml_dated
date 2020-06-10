# ------------------------------------------------------------
# Normal shock structure
# ------------------------------------------------------------

using Revise
using OrdinaryDiffEq
using Flux
using DiffEqFlux
using Optim
using Plots
using FileIO
using JLD2
using OffsetArrays
using ProgressMeter

#using Kinetic
include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
using .Kinetic

# load variables
cd(@__DIR__)
D = read_dict("fsm3d.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

γ = heat_capacity_ratio(inK, 3)
set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)
μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = GasProperty(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)
vSpace = VSpace1D(umin, umax, nu, vMeshType)
vSpace2D = VSpace2D(vmin, vmax, nv, wmin, wmax, nw, vMeshType)
vSpace3D = VSpace3D(umin, umax, nu, vmin, vmax, nv, wmin, wmax, nw, vMeshType)
wL, primL, fL, bcL, wR, primR, fR, bcR = ib_rh(mach, γ, vSpace3D.u, vSpace3D.v, vSpace3D.w)
ib = IB1D1F(wL, primL, fL, bcL, wR, primR, fR, bcR)
ks = SolverSet(set, pSpace, vSpace3D, gas, ib, pwd())

kn_bzm = hs_boltz_kn(ks.gas.μᵣ, 1.0)
sos = sound_speed(ks.ib.primR, γ)
tmax = (ks.vSpace.u1 + sos) / ks.pSpace.dx[1]
dt = Float32(ks.set.cfl / tmax)
tspan = (0.f0, dt)
tran = range(tspan[1], tspan[2], length = tLen)

ctr = OffsetArray{ControlVolume1D1F}(undef, eachindex(ks.pSpace.x))
face = Array{Interface1D1F}(undef, ks.pSpace.nx + 1)
for i in eachindex(ctr)
    if i <= ks.pSpace.nx ÷ 2
        ctr[i] = ControlVolume1D1F(
            ks.pSpace.x[i],
            ks.pSpace.dx[i],
            Float32.(ks.ib.wL),
            Float32.(ks.ib.primL),
            Float32.(ks.ib.fL),
        )
    else
        ctr[i] = ControlVolume1D1F(
            ks.pSpace.x[i],
            ks.pSpace.dx[i],
            Float32.(ks.ib.wR),
            Float32.(ks.ib.primR),
            Float32.(ks.ib.fR),
        )
    end
end
for i = 1:ks.pSpace.nx+1
    face[i] = Interface1D1F(ks.ib.wL, ks.ib.fL)
end

phi, psi, phipsi = kernel_mode(
    nm,
    vSpace3D.u1,
    vSpace3D.v1,
    vSpace3D.w1,
    vSpace3D.du[1, 1, 1],
    vSpace3D.dv[1, 1, 1],
    vSpace3D.dw[1, 1, 1],
    vSpace3D.nu,
    vSpace3D.nv,
    vSpace3D.nw,
    alphaRef,
)

function boltzmann(f, p, t)
    Kn, M, phi, psi, phipsi = p
    return boltzmann_fft(f, Kn, M, phi, psi, phipsi)
end

function bgk(f, p, t)
    M, tau = p
    return (M .- f) ./ tau
end

function step_be!(
    fwL,
    ffL,
    cons,
    prim,
    f,
    fwR,
    ffR,
    γ,
    μ,
    ω,
    u,
    v,
    w,
    weights,
    dx,
)

    #M = maxwellian(u, v, w, prim)
    #tau = vhs_collision_time(prim, μ, ω)
    #df = (M .- f) ./ tau

    #df = boltzmann(f, [kn_bzm, nm, phi, psi, phipsi], tspan)
    #f_star = f #.+ df .* dt

    prob = ODEProblem(boltzmann, f[1:end, 1:end, 1:end], tspan, [kn_bzm, nm, phi, psi, phipsi])
    f_star = Array(solve(prob, Euler(), dt = dt))[:, :, :, end]

    @. f = f_star + (ffL - ffR) / dx

    #--- update W^{n+1} ---#
    @. cons += (fwL - fwR) / dx
    prim .= conserve_prim(cons, γ)

end

@showprogress for iter = 1:100
    Kinetic.evolve!(ks, ctr, face, dt)

    Threads.@threads for i = 2:49
        step_be!(
            face[i].fw,
            face[i].ff,
            ctr[i].w,
            ctr[i].prim,
            ctr[i].f,
            face[i+1].fw,
            face[i+1].ff,
            ks.gas.γ,
            ks.gas.μᵣ,
            ks.gas.ω,
            ks.vSpace.u,
            ks.vSpace.v,
            ks.vSpace.w,
            ks.vSpace.weights,
            ctr[i].dx,
        )
    end
end

plot_line(ks, ctr)

boltzmann(f0, [kn_bzm, nm, phi, psi, phipsi], tspan)

M = maxwellian(vSpace3D.u, vSpace3D.v, vSpace3D.w, ctr[1].prim)
tau = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
(M .- ctr[1].f) ./ tau




f0 =
    Float32.(
        0.5 * (1 / π)^1.5 .*
        (exp.(-(vSpace3D.u .- 0.99) .^ 2) .+ exp.(-(vSpace3D.u .+ 0.99) .^ 2)) .*
        exp.(-vSpace3D.v .^ 2) .* exp.(-vSpace3D.w .^ 2),
    ) |> Array








plot(vSpace3D.u[:,1,1],boltzmann(f0, [kn_bzm, nm, phi, psi, phipsi], tspan)[:,12,12])
