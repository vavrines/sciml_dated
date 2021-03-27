using Kinetic, PyPlot, DataFrames
using KitBase.Plots, KitBase.JLD2
using KitML.CSV

cd(@__DIR__)
mc3 = CSV.File("../dsmc_rarefied.csv") |> DataFrame

begin
    set = Setup(
        "gas", # matter
        "creep", # case
        "2d2f2v", # space
        "kfvs", # flux
        "shakhov", # collision: for scalar conservation laws there are none
        1, # species
        2, # interpolation order
        "vanleer", # limiter
        "maxwell", # boundary
        0.8, # cfl
        100.0, # simulation time
    )

    ps = PSpace2D(0.0, 5.0, 200, 0.0, 1.0, 40)
    vs = VSpace2D(-5.0, 5.0, 40, -5.0, 5.0, 40)
    Kn = 0.064
    gas = KitBase.Gas(Kn, 0.0, 2/3, 1, 5/3, 0.81, 1.0, 0.5, ref_vhs_vis(Kn, 1.0, 0.5))

    prim0 = [1.0, 0.0, 0.0, 1.0]
    w0 = prim_conserve(prim0, 5/3)
    h0 = maxwellian(vs.u, vs.v, prim0)
    b0 = @. h0 * gas.K / (2 * prim0[end])
    bcL = deepcopy(prim0)
    bcR = [1.0, 0.0, 0.0, 273/573]
    ib = IB2F(w0, prim0, h0, b0, bcL, w0, prim0, h0, b0, bcR)

    ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)
    cd(@__DIR__)
end

begin
    @load "ctr.jld2" ctr
    field = zeros(ks.pSpace.nx, ks.pSpace.ny, ks.vSpace.nu, ks.vSpace.nv)
    for j in axes(field, 2), i in axes(field, 1)
        field[i, j, :, :] .= ctr[i, j].h
    end
end



PyPlot.contour3d


PyPlot.contourf(ks.pSpace.y[1, 1:end], ks.vSpace.u[1:end, 1], ks.vSpace.v[1, 1:end], field[end÷2, :, :, :]', linewidth=1, levels=20, cmap=ColorMap("inferno"))





begin
    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    #PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field3[:, :, 4]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    PyPlot.contourf(ks.pSpace.y[1, 1:end], ks.vSpace.u[1:end, 1], ks.vSpace.v[1, 1:end], field[end÷2, :, :, :]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    colorbar()
    #colorbar(orientation="horizontal")
    #PyPlot.streamplot(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field3[:, :, 2]', field3[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
    #xlabel("x")
    #ylabel("y")
    #PyPlot.title("U-velocity")
    #xlim(0.01, 4.99)
    #ylim(0.01, 0.99)
    #PyPlot.axes().set_aspect(1.2)
    #PyPlot.grid("on")
    display(fig)
    #fig.savefig("cavity_u.pdf")
end

Plots.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field3[:, :, 2]')
Plots.plot(ks.pSpace.x[1:end, 1], field3[:, 20, 2])

Plots.plot(ks.pSpace.x[1:end, 1], field3[:, 20, 1] .* field3[:, 20, 4])
Plots.scatter!(mc3.x, mc3.Curve3 ./ 101.325)