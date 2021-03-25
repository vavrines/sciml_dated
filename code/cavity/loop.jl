using Kinetic
using KitBase.ProgressMeter, KitBase.JLD2
using KitML.Flux, KitML.DiffEqFlux, KitBase.Plots
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

function step(
    w::T1,
    prim::T1,
    h::T2,
    fwL::T1,
    fhL::T2,
    fwR::T1,
    fhR::T2,
    fwD::T1,
    fhD::T2,
    fwU::T1,
    fhU::T2,
    u::T2,
    v::T2,
    weights::T2,
    p,
    mode = :bgk,
) where {
    T1<:AbstractArray{<:AbstractFloat,1},
    T2<:AbstractArray{<:AbstractFloat,2},
}

    γ, μᵣ, ω, Pr, Δs, dt, RES, AVG = p[1:8]
    ann = p[9:end]

    #--- record W^{n} ---#
    w_old = deepcopy(w)
    h_old = deepcopy(h)

    H_old = maxwellian(u, v, prim)
    τ_old = vhs_collision_time(prim, μᵣ, ω)

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR + fwD - fwU) / Δs
    prim .= conserve_prim(w, γ)

    H = maxwellian(u, v, prim)
    τ = vhs_collision_time(prim, μᵣ, ω)

    #--- update f^{n+1} ---#
    for j in axes(v, 2), i in axes(u, 1)
        h[i, j] = (h[i, j] + (fhL[i, j] - fhR[i, j] + fhD[i, j] - fhU[i, j]) / Δs + dt / τ * H[i, j]) / (1.0 + dt / τ)
    end

    if mode == :bgk
        @. h = (h + H / τ * dt) / (1.0 + dt / τ)
    elseif mode == :shakhov
        H_old = maxwellian(u, v, prim)
        qf = heat_flux(h, prim, u, v, weights)
        H1 = shakhov(u, v, H_old, qf, prim, Pr)
        H .+= H1

        @. h = (h + H / τ * dt) / (1.0 + dt / τ)
    elseif mode == :nn
        df = ube_dfdt(h_old[:], (H_old[:], τ_old, ann), dt)
        df = reshape(df, size(h)) .+ (H_old .- h) / τ_old
        @. h += df * dt
    end

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

end

function update(
    KS::X,
    ctr::Y,
    a1face::Z,
    a2face::Z,
    dt,
    residual,
    nn,
    para;
    coll = :bgk::Symbol,
) where {
    X<:AbstractSolverSet,
    Y<:AbstractArray{ControlVolume2D1F,2},
    Z<:AbstractArray{Interface2D1F,2},
}
    sumRes = zero(KS.ib.wL)
    sumAvg = zero(KS.ib.wL)

    @inbounds for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx
            step(
                ctr[i, j].w, ctr[i, j].prim, ctr[i, j].f,
                a1face[i, j].fw,
                a1face[i, j].ff,
                a1face[i+1, j].fw,
                a1face[i+1, j].ff,
                a2face[i, j].fw,
                a2face[i, j].ff,
                a2face[i, j+1].fw,
                a2face[i, j+1].ff,
                KS.vSpace.u,
                KS.vSpace.v,
                KS.vSpace.weights,
                (KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ω,
                KS.gas.Pr,
                ctr[i, j].dx * ctr[i, j].dy,
                dt,
                sumRes,
                sumAvg,
                nn,
                para),
                coll,
            )
        end
    end

    for i in eachindex(residual)
        residual[i] = sqrt(sumRes[i] * KS.pSpace.nx * KS.pSpace.ny) / (sumAvg[i] + 1.e-7)
    end

    return nothing
end

residual = zeros(4)
dt = timestep(ks, ctr, t)
nt = (ks.set.maxTime ÷ dt) |> Int
@showprogress for iter = 1:5#nt
    Kinetic.reconstruct!(ks, ctr)
    Kinetic.evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    #Kinetic.update!(ks, ctr, a1face, a2face, dt, residual; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))
    update(ks, ctr, a1face, a2face, dt, residual, nn, u; coll = :nn)

    global t += dt
end
plot_contour(ks, ctr)
#@time ube_dfdt(ctr[1,1].f[:], (ctr[1,1].f[:], 0.1, (nn,u)), dt)

# use central point for benchmark
i = 23
j = 23
_m = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
_tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
## shakhov
@benchmark begin
    _m = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
    _tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
    _q = heat_flux(ctr[i, j].f, ctr[i, j].prim, ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights)
    _s = _m .+ shakhov(ks.vSpace.u, ks.vSpace.v, _m, _q, ctr[i, j].prim, ks.gas.Pr)
    _q = (_s - ctr[i, j].f) ./ _tau
end
## nn
@benchmark begin
    nn((_m .- ctr[i, j].f)[:], u)
end
## fsm
v3d = VSpace3D(ks.vSpace.u0, ks.vSpace.u1, ks.vSpace.nu, 
    ks.vSpace.v0, ks.vSpace.v1, ks.vSpace.nv, 
    ks.vSpace.v0, ks.vSpace.v1, ks.vSpace.nv, "rectangle")
phi, psi, phipsi = kernel_mode(8, v3d.u1, v3d.v1, v3d.w1, v3d.du[1,1,1], v3d.dv[1,1,1], v3d.dw[1,1,1],
    v3d.nu, v3d.nv, v3d.nw, 1.0)
@benchmark begin
    f0 = zeros(v3d.nu, v3d.nv, v3d.nw)
    for i3 in axes(f0, 3), i2 in axes(f0, 2), i1 in axes(f0, 1)
        f0[i1, i2, i3] = ctr[i, j].f[i1, i2] * (ctr[i, j].prim[end] / π)^0.5 * exp(-ctr[i, j].prim[end] * v3d.w[i1, i2, i3]^2)
    end
    boltzmann_fft(f0, ks.gas.Kn, 8, phi, psi, phipsi)
end