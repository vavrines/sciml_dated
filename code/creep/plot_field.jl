using Kinetic, PyPlot, DataFrames
using KitBase.Plots, KitBase.JLD2
using KitML.CSV

cd(@__DIR__)
mc1 = CSV.File("dsmc_continuum.csv") |> DataFrame
mc2 = CSV.File("dsmc_transition.csv") |> DataFrame
mc3 = CSV.File("dsmc_rarefied.csv") |> DataFrame

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
    vs = VSpace2D(-5.0, 5.0, 28, -5.0, 5.0, 28)
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
    @load "continuum/ctr.jld2" ctr
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field1[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
    @load "transition/ctr.jld2" ctr
    field2 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field2, 2), i in axes(field2, 1)
        field2[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field2[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
    #@load "rarefied/ctr.jld2" ctr
    @load "finemesh/ctr.jld2" ctr
    field3 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field3, 2), i in axes(field3, 1)
        field3[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field3[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
end

begin
    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field3[:, :, 4]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    #colorbar()
    colorbar(orientation="horizontal")
    PyPlot.streamplot(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field3[:, :, 2]', field3[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
    xlabel("x")
    ylabel("y")
    #PyPlot.title("U-velocity")
    xlim(0.01, 4.99)
    ylim(0.01, 0.99)
    PyPlot.axes().set_aspect(1.2)
    #PyPlot.grid("on")
    display(fig)
    #fig.savefig("creep_kn3.pdf")
end

Plots.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field3[:, :, 2]')
Plots.plot(ks.pSpace.x[1:end, 1], field3[:, 20, 2])

Plots.plot(ks.pSpace.x[1:end, 1], field1[:, 20, 1] .* field1[:, 20, 4], lw=2, label="Kn=0.064 (UBE)", xlabel="x", ylabel="p/p₀", legend=:topleft)
Plots.plot!(ks.pSpace.x[1:end, 1], field2[:, 20, 1] .* field2[:, 20, 4], lw=2, label="Kn=0.64 (UBE)")
Plots.plot!(ks.pSpace.x[1:end, 1], field3[:, 20, 1] .* field3[:, 20, 4], lw=2, label="Kn=3.2 (UBE)")
Plots.scatter!(mc1.x[1:2:end], mc1.Curve1[1:2:end] ./ 101.325, color=3, label="Kn=0.064 (DSMC)")
Plots.scatter!(mc2.x[1:2:end], mc2.Curve2[1:2:end] ./ 101.325, color=5, label="Kn=0.64 (DSMC)")
Plots.scatter!(mc3.x[1:2:end], mc3.Curve3[1:2:end] ./ 101.325, color=13, label="Kn=3.2 (DSMC)")
Plots.savefig("creep_pressure.pdf")