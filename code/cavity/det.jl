using OrdinaryDiffEq, CUDA, Kinetic
using KitBase.ProgressMeter
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
pc = initial_params(nn) |> CuArray

residual = zeros(4)
dt = timestep(ks, ctr, t)
nt = (ks.set.maxTime ÷ dt) |> Int
@showprogress for iter = 1:200#nt
    Kinetic.reconstruct!(ks, ctr)
    Kinetic.evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    Kinetic.update!(ks, ctr, a1face, a2face, dt, residual; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))
    global t += dt
end
plot_contour(ks, ctr)

F = zeros(Float32, n0, ks.pSpace.nx*ks.pSpace.ny)
for i = 1:ks.pSpace.nx, j = 1:ks.pSpace.ny
    idx = ks.pSpace.ny * (i - 1) + j
    F[:, idx] .= ctr[i, j].f[:]
end
Fc = F |> CuArray

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

#res = sci_train(loss, pc, ADAM(); cb=Flux.throttle(cb, 1), maxiters=200)
res = sci_train(loss, res.u, ADAM(); cb=Flux.throttle(cb, 1), maxiters=1000)

_t1 = reshape(nn(Xc[:, 400], res.u), 16, 16)
_t2 = reshape(Yc[:, 400], 16, 16)
contourf(_t1 |> Array)
contourf(_t2 |> Array)


_t = reshape(S[:, 400] .- M[:, 400], 16, 16)
contourf(_t)

_m = maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[23, 23].prim)
_tau = vhs_collision_time(ctr[23, 23].prim, ks.gas.μᵣ, ks.gas.ω)
_q = heat_flux(ctr[23, 23].f, ctr[23, 23].prim, ks.vSpace.u, ks.vSpace.v, ks.vSpace.weights)
_s = shakhov(ks.vSpace.u, ks.vSpace.v, _m, _q, ctr[23, 23].prim, ks.gas.Pr)

_r1 = (_s .- ctr[23, 23].f)/_tau
_r2 = reshape(nn((_m .- ctr[23,23].f)[:], res.u |> cpu), ks.vSpace.nu, :)

contourf(_s ./ _tau)
contourf(_r2)


contourf((_m .- ctr[23,23].f) /_tau)
contourf((_m .+ _s .- ctr[23,23].f) /_tau)


