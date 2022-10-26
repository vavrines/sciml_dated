using KitBase
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using Plots

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")

res = zeros(4)
dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int
@showprogress for iter = 1:500#nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, a1face, a2face, dt)
    update!(ks, ctr, a1face, a2face, dt, res)
end

plot(ks, ctr)

sol = zeros(ks.ps.nx, ks.ps.ny, 4)
for i in axes(sol, 1), j in axes(sol, 2)
    sol[i, j, :] .= ctr[i, j].prim
end

@save "det.jld2" ks ctr
