using OrdinaryDiffEq, Kinetic
using KitBase.ProgressMeter
using KitML.DiffEqFlux

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")

res = zeros(4)
dt = timestep(ks, ctr, t)
nt = (ks.set.maxTime ÷ dt) |> Int
@showprogress for iter = 1:200#nt
    Kinetic.reconstruct!(ks, ctr)
    Kinetic.evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    Kinetic.update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))
    global t += dt
end

plot_contour(ks, ctr)



n0 = ks.vSpace.nu * ks.vSpace.nu
nh = 8

nn = FastChain(FastDense(n0, n0*nh, tanh), FastDense(n0*nh, n0*nh, tanh), FastDense(n0*nh, n0))
p = initial_params(nn)

X = zeros(Float32, n0, ks.pSpace.nx*ks.pSpace.ny)
for i = 1:ks.pSpace.nx, j = 1:ks.pSpace.ny
    idx = ks.pSpace.ny * (i - 1) + j
    X[:, idx] .= ctr[i, j].f[:]
end

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
P = [M, τ]

prob = ODEProblem(bgk_ode!, X, (0.0, dt), (S, τ))
Y = solve(prob, Midpoint(), saveat=0:dt/2:dt) |> Array

function dfdt(df, f, p, t)
    df .= ((M .- f) ./ τ .+ nn(f, p))
end

reshape(M[:, 1], ks.vSpace.nu, ks.vSpace.nv)