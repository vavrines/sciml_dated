using Kinetic
using KitBase.ProgressMeter, KitBase.JLD2, KitBase.Plots
using KitML.Flux, KitML.DiffEqFlux, KitML.Optim

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")

n0 = ks.vSpace.nu * ks.vSpace.nu
nh = 8

nn = FastChain(
    FastDense(n0, n0*nh, tanh),
    #FastDense(n0*nh, n0*nh, tanh),
    FastDense(n0*nh, n0),
)

@load "det.jld2" ctr
plot_contour(ks, ctr)

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
X = Array(Xc)

dM = zero(Mc)
dS = zero(Sc)
bgk_ode!(dM, Fc, (Mc, τc), 0.0)
bgk_ode!(dS, Fc, (Sc, τc), 0.0)

Yc = dS .- dM
Y = Array(Yc)

function loss_gpu(p)
    sol = nn(Xc, p)
    return sum(abs2, sol .- Yc)
end

function loss_cpu(p)
    sol = nn(X, p)
    return sum(abs2, sol .- Y)
end

cb = function (p, l)
    println("loss: $l")
    return false
end

pc = initial_params(nn) |> gpu
@load "para_nn.jld2" u
loss_cpu(u)

res = sci_train(loss_gpu, pc, ADAM(); cb=Flux.throttle(cb, 1), maxiters=200)
res = sci_train(loss_gpu, gpu(u), ADAM(1e-4); cb=Flux.throttle(cb, 1), maxiters=250)
res = sci_train(loss_gpu, res.u, ADAM(1e-4); cb=Flux.throttle(cb, 1), maxiters=280, save_best=true)

u = res.u |> cpu
@save "para_nn.jld2" u