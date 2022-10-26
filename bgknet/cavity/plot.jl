using KitBase, CairoMakie, NipponColors, DataFrames
using KitBase.JLD2, KitBase.CSV

cd(@__DIR__)
dc = dict_color()

ks = initialize("config.txt")[1]
JLD2.@load "shakhov.jld2" ctr
vline_dsmc = CSV.File("../../code/cavity/dsmc_vline.csv") |> DataFrame
hline_dsmc = CSV.File("../../code/cavity/dsmc_hline.csv") |> DataFrame

sol = zeros(ks.ps.nx, ks.ps.ny, 6)
for i in axes(sol, 1), j in axes(sol, 2)
	sol[i, j, 1:3] .= ctr[i, j].prim[1:3]
	sol[i, j, 4] = 1 / ctr[i, j].prim[4]
	sol[i, j, 5:6] .= heat_flux(ctr[i, j].h, ctr[i, j].b, ctr[i, j].prim,
	ks.vs.u, ks.vs.v, ks.vs.weights)
end

x = ks.ps.x[:, 1]
y = ks.ps.y[1, :]

interp(X, u) = begin
   x, y = X
   f1 = u[1]([x, y]) |> first
   f2 = u[2]([x, y]) |> first
   Point2f(f1, f2)
end

ucurves = [
   KB.itp.RegularGridInterpolator((x, y), sol[:, :, 2]),
   KB.itp.RegularGridInterpolator((x, y), sol[:, :, 3]),
]
interp(X) = interp(X, ucurves)

begin
   fig = Figure()
   ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "", aspect = 1)
   co = contourf!(ks.ps.x[:, 1], ks.ps.y[1, :], sol[:, :, 1]; colormap = :PiYG_8, levels = 15)
   Colorbar(fig[1, 2], co)
   st = streamplot!(interp, x[1]..x[end], y[1]..y[end]; colormap = :blues)
   fig
end
save("cavity_density.pdf", fig)

qcurves = [
   KB.itp.RegularGridInterpolator((x, y), sol[:, :, 5]),
   KB.itp.RegularGridInterpolator((x, y), sol[:, :, 6]),
]
interp(X) = interp(X, qcurves)

begin
   fig = Figure()
   ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "", aspect = 1)
   co = contourf!(ks.ps.x[:, 1], ks.ps.y[1, :], sol[:, :, 4]; colormap = :PiYG_8, levels = 15)
   Colorbar(fig[1, 2], co)
   st = streamplot!(interp, x[1]..x[end], y[1]..y[end]; colormap = :blues)
   fig
end
save("cavity_temperature.pdf", fig)

vline = zeros(size(ctr, 2))
for i in eachindex(vline)
    vline[i] = ctr[23, i].prim[2] / 0.15
end
hline = zeros(size(ctr, 1))
for i in eachindex(hline)
    hline[i] = ctr[i, 23].prim[3] / 0.15
end

begin
   fig = Figure()
   ax = Axis(fig[1, 1], xlabel = "u/Uw", ylabel = "y", title = "")
   lines!(vline, ks.ps.y[23, :]; color = dc["ro"], label = "UBE")
   scatter!(vline_dsmc.u, vline_dsmc.y; color = (dc["tokiwa"], 0.7), label = "DSMC")
   axislegend()
   fig
end
save("cavity_vline.pdf", fig)

begin
   fig = Figure()
   ax = Axis(fig[1, 1], xlabel = "x", ylabel = "v/Uw", title = "")
   lines!(ks.ps.x[:, 23], hline; color = dc["ro"], label = "UBE")
   scatter!(hline_dsmc.x, hline_dsmc.v; color = (dc["tokiwa"], 0.7), label = "DSMC")
   axislegend()
   fig
end
save("cavity_hline.pdf", fig)
