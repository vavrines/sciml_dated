using Kinetic, PyPlot, DataFrames
using KitBase.Plots, KitBase.JLD2
using KitML.CSV

cd(@__DIR__)
href1 = CSV.File("reference/hline_kn0.08_argon.csv") |> DataFrame
href2 = CSV.File("reference/hline_kn10.csv") |> DataFrame
#=
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

    #ps = PSpace2D(0.0, 5.0, 300, 0.0, 1.0, 60)
    #ps = PSpace2D(0.0, 5.0, 200, 0.0, 1.0, 40)
    ps = PSpace2D(0.0, 5.0, 100, 0.0, 1.0, 20)
    vs = VSpace2D(-4.5, 4.5, 28, -4.5, 4.5, 28, "rectangle")
    #vs = VSpace2D(-5.0, 5.0, 36, -5.0, 5.0, 36, "algebra")
    Kn = 10.0
    gas = KitBase.Gas(Kn, 0.0, 2/3, 1, 5/3, 0.81, 1.0, 0.5, ref_vhs_vis(Kn, 1.0, 0.5))

    prim0 = [1.0, 0.0, 0.0, 1.0]
    w0 = prim_conserve(prim0, 5/3)
    h0 = maxwellian(vs.u, vs.v, prim0)
    b0 = @. h0 * gas.K / (2 * prim0[end])
    bcL = deepcopy(prim0)
    bcR = [1.0, 0.0, 0.0, 0.5]
    ib = IB2F(w0, prim0, h0, b0, bcL, w0, prim0, h0, b0, bcR)

    ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)
    cd(@__DIR__)
end
=#
begin
    #@load "rarefied/mid/t3/ctr.jld2" ctr
    #@load "data/ctr100_alg48_kn0.08.jld2" ctr
    #@load "rarefied/mesh200/quad72/ctr.jld2" ctr
    #@load "continuum/mesh300/ctr.jld2" ctr
    #@load "continuum/mesh200/ctr.jld2" ctr
    #@load "continuum/mesh200/sol.jld2" ks ctr
    #@load "continuum/mesh200/sol.jld2" ks ctr
    @load "continuum/mesh100 down/sol.jld2" ks ctr
    #@load "c2/ctr.jld2"
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field1[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
end

begin
    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field1[:, :, 2]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    #colorbar()
    colorbar(orientation="horizontal")
    PyPlot.streamplot(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field1[:, :, 2]', field1[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
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

Plots.plot(ks.pSpace.x[1:end, 1], field1[:, end÷2+1, 2])
Plots.plot!(href1.x, href1.u)
Plots.plot!(ks.pSpace.x[1:end, 1], field1[:, end÷2+1, 2])



Plots.plot(ks.pSpace.x[1:end, 1], (field1[:, end÷2, 2] .+ field1[:, end÷2+1, 2])./2.)
Plots.plot!(href1.x.*1.02.-0.08, href1.u.+0.0003)


Plots.plot!(ks.pSpace.x[1:end, 1], (field1[:, end÷2, 2] .+ field1[:, end÷2+1, 2])./2)

Plots.plot(href2.x, href2.u)

Plots.contourf(ks.vSpace.u[1:end, 1], ks.vSpace.v[1, 1:end], ctr[200, 2].h[1:end, 1:end])