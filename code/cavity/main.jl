using OrdinaryDiffEq, Kinetic
using KitBase.ProgressMeter, KitBase.JLD2
using KitML.Flux, KitML.DiffEqFlux, KitBase.Plots

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")

n0 = ks.vSpace.nu * ks.vSpace.nu
nh = 8

nn = FastChain(
    FastDense(n0, n0*nh, tanh),
    FastDense(n0*nh, n0*nh, tanh),
    FastDense(n0*nh, n0),
)
pc = initial_params(nn) |> gpu

#=residual = zeros(4)
dt = timestep(ks, ctr, t)
nt = (ks.set.maxTime ÷ dt) |> Int
@showprogress for iter = 1:200#nt
    Kinetic.reconstruct!(ks, ctr)
    Kinetic.evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    Kinetic.update!(ks, ctr, a1face, a2face, dt, residual; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))
    global t += dt
end
plot_contour(ks, ctr)=#
@load "det.jld2" ctr

F = zeros(Float32, n0, ks.pSpace.nx*ks.pSpace.ny)
for i = 1:ks.pSpace.nx, j = 1:ks.pSpace.ny
    idx = ks.pSpace.ny * (i - 1) + j
    F[:, idx] .= ctr[i, j].f[:]
end
Fc = F |> gpu

M = Array{Float32}(undef, n0, ks.pSpace.nx*ks.pSpace.ny)
S = Array{Float32}(undef, n0, ks.pSpace.nx*ks.pSpace.ny)
τ = Array{Float32}(undef, 1, ks.pSpace.nx*ks.pSpace.ny)
Threads.@threads for j = 1:ks.pSpace.ny
    for i = 1:ks.pSpace.nx
        idx = ks.pSpace.ny * (i - 1) + j
        M[:, idx] .= maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)[:]
        q = heat_flux(ctr[i, j].f, ctr[i, j].prim, ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights)
        S[:, idx] .= M[:, idx] .+ shakhov(ks.vSpace.u, ks.vSpace.v, reshape(M[:, idx], ks.vSpace.nu, ks.vSpace.nv), q, ctr[i, j].prim, ks.gas.Pr)[:]
        τ[1, idx] = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
    end
end
Mc = gpu(M)
Sc = gpu(S)
τc = gpu(τ)

Xc = Mc .- Fc

dM = zero(Mc)
dS = zero(Sc)
bgk_ode!(dM, Fc, (Mc, τc), 0.0)
bgk_ode!(dS, Fc, (Sc, τc), 0.0)

Yc = dS .- dM

function loss(p)
    sol = nn(Xc, p)
    return sum(abs2, sol .- Yc)
end

cb = function (p, l)
    println("loss: $l")
    return false
end

res = sci_train(loss, pc, ADAM(); cb=Flux.throttle(cb, 1), maxiters=200)
res = sci_train(loss, res.u, ADAM(1e-4); cb=Flux.throttle(cb, 1), maxiters=1000)

i = 1000
_t1 = reshape(nn(Xc[:, i], res.u), ks.vSpace.nu, :)
_t2 = reshape(Yc[:, i], ks.vSpace.nu, :)
contourf(_t1 |> Array)
contourf(_t2 |> Array)

#u = res.u |> cpu
#@save "para_nn.jld2" u

i = 4; j = 4
_m = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
_tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)
_q = heat_flux(ctr[i, j].f, ctr[i, j].prim, ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights)
_s = shakhov(ks.vSpace.u, ks.vSpace.v, _m, _q, ctr[i, j].prim, ks.gas.Pr)

_r1 = _s /_tau
_r2 = reshape(nn((_m .- ctr[i, j].f)[:], res.u |> cpu), ks.vSpace.nu, :)

contourf(_s ./ _tau)
contourf(_r2)



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
    p;
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

dt = timestep(ks, ctr, t)
step(ctr[i,j].w, ctr[i,j].prim, ctr[i,j].f,
    a1face[i,j].fw,a1face[i,j].ff,a1face[i+1,j].fw,a1face[i+1,j].ff,
    a2face[i,j].fw,a2face[i,j].ff,a2face[i,j+1].fw,a2face[i,j+1].ff,
    ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights,
    (ks.gas.γ, ks.gas.μᵣ, ks.gas.ω, ks.gas.Pr, ctr[i,j].dx*ctr[i,j].dy, dt, zeros(4), zeros(4), nn, cpu(res.u)),
    mode=:nn)
