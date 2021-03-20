using OrdinaryDiffEq, CUDA, Kinetic
using KitBase.ProgressMeter
using KitML.Flux, KitML.DiffEqFlux

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
p = initial_params(nn) |> CuArray

X = zeros(Float32, n0, ks.pSpace.nx*ks.pSpace.ny)
for i = 1:ks.pSpace.nx, j = 1:ks.pSpace.ny
    idx = ks.pSpace.ny * (i - 1) + j
    X[:, idx] .= ctr[i, j].f[:]
end
XC = CuArray(X)

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
MC = CuArray(M)
SC = CuArray(S)
τC = CuArray(τ)

prob = ODEProblem(bgk_ode!, XC, (0.0, dt), (SC, τC))
Y = solve(prob, Euler(), dt=dt).u[end]

function dfdt(df, f, p, t)
    df .= ((MC .- f) ./ τC .+ nn(f, p))
end

prob_ube = ODEProblem(dfdt, XC, (0.0, dt), p)
function loss(p)
    #sol_ube = solve(prob_ube, Midpoint(), u0=X, p=p, saveat=0:dt/2:dt)
    sol_ube = solve(prob_ube, Euler(), u0=XC, p=p, dt=dt)
    
    return sum(abs2, sol_ube.u[end] .- Y)
end

cb = function (p, l)
    println("loss: $l")
    return false
end

res = sci_train(loss, p, ADAM(); cb=Flux.throttle(cb, 1), maxiters=200)





pc = res.u |> CuArray
XC = CuArray(X)
YC = CuArray(Y)

prob_cuda = ODEProblem(dfdt, XC, (0.0, dt), pc)
function loss(p)
    sol = solve(prob_cuda, Euler(), u0=XC, p=pc, dt=dt)
    return sum(abs2, Array(sol) .- YC)
end

solve(prob_cuda, Euler(), u0=XC, p=pc, dt=dt)