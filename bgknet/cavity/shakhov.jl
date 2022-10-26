using KitBase
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
ks, ctr, a1face, a2face, t = initialize("config.txt")

res = zeros(4)
dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int
@showprogress for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, a1face, a2face, dt)
    update!(ks, ctr, a1face, a2face, dt, res)
end

@save "shakhov.jld2" ks ctr
