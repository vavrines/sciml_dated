using Kinetic, PyPlot
using KitBase.Plots, KitBase.JLD2

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

@load "ctr.jld2" ctr

begin
    close("all")
    field = zeros(ks.pSpace.nx, ks.pSpace.ny, 3)
    for j in axes(field, 2), i in axes(field, 1)
        field[i, j, 1:2] .= ctr[i, j].prim[2:3]
        field[i, j, 3] = 1 / ctr[i, j].prim[end]
    end
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field[:, :, 3]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    colorbar()
    PyPlot.streamplot(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field[:, :, 1]', field[:, :, 2]', density=1.3, color="moccasin", linewidth=1)
    xlabel("x")
    ylabel("y")
    #PyPlot.title("U-velocity")
    xlim(0.01,4.99)
    ylim(0.01,0.99)
    #PyPlot.grid("on")
    display(fig)
    #fig.savefig("cavity_u.pdf")
end

Plots.contourf(field[:, :, 1])