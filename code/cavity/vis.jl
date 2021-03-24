using DataFrames, KitBase, KitBase.CSV, KitBase.Plots, KitBase.JLD2, PyPlot

cd(@__DIR__)
JLD2.@load "det.jld2" ks ctr
vline_dsmc = CSV.File("dsmc_vline.csv") |> DataFrame
hline_dsmc = CSV.File("dsmc_hline.csv") |> DataFrame

vline = zeros(size(ctr, 2))
for i in eachindex(vline)
    vline[i] = ctr[23, i].prim[2] / 0.15
end
hline = zeros(size(ctr, 1))
for i in eachindex(hline)
    hline[i] = ctr[i, 23].prim[3] / 0.15
end

Plots.plot(vline, ks.pSpace.y[23, :], lw=2, label="UBE", xlabel="U/Uw", ylabel="y")
Plots.scatter!(vline_dsmc.u, vline_dsmc.y, label="DSMC")
Plots.savefig("vline.pdf")

Plots.plot(ks.pSpace.x[:, 23], hline, lw=2, label="UBE", xlabel="x", ylabel="V/Uw")
Plots.scatter!(hline_dsmc.x, hline_dsmc.v, label="DSMC")
Plots.savefig("hline.pdf")