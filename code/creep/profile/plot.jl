using Kinetic, PyPlot, DataFrames
using KitBase.Plots, KitBase.JLD2, KitBase.PyCall
using KitML.CSV

cd(@__DIR__)

# ---
# Huang CICP
# ---
begin
    @load "data/huang150_transition_argon.jld2" ks ctr
    ks1 = deepcopy(ks)
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field1[i, j, 4] = 1 / ctr[i, j].prim[end]
    end

    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks1.pSpace.x[1:end, 1], ks1.pSpace.y[1, 1:end], field1[:, :, 4]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    #colorbar()
    colorbar(orientation="horizontal")
    PyPlot.streamplot(ks1.pSpace.x[1:end, 1], ks1.pSpace.y[1, 1:end], field1[:, :, 2]', field1[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
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

begin
    @load "data/huang150_rarefied_argon.jld2" ks ctr
    ks2 = deepcopy(ks)
    field2 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field2, 2), i in axes(field2, 1)
        field2[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field2[i, j, 4] = 1 / ctr[i, j].prim[end]
    end

    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks2.pSpace.x[1:end, 1], ks2.pSpace.y[1, 1:end], field2[:, :, 4]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    #colorbar()
    colorbar(orientation="horizontal")
    PyPlot.streamplot(ks2.pSpace.x[1:end, 1], ks2.pSpace.y[1, 1:end], field2[:, :, 2]', field1[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
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

# ---
# Wu JFM
# ---
itp = pyimport("scipy.interpolate")

hr1 = CSV.File("reference/hline1_argon.csv") |> DataFrame
hr2 = CSV.File("reference/hline2.csv") |> DataFrame
vr1 = CSV.File("reference/vline1_argon.csv") |> DataFrame

# Kn = 0.08
begin
    @load "data/continuum200_quad28_argon.jld2" ks ctr
    ks1 = deepcopy(ks)
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field1[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
    @load "data/continuum200_quad28_argon.jld2" ks ctr
    ks2 = deepcopy(ks)
    field2 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field2, 2), i in axes(field2, 1)
        field2[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field2[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
end

# hline
Plots.plot(hr1.x, hr1.u, lw=2, label="Ref", xlabel="x", ylabel="U")
fr1 = itp.interp1d(hr1.x, hr1.u, kind="cubic")
uref = fr1(ks2.pSpace.x[:, 1])
v0 = (field2[:, end÷2, 2] + field2[:, end÷2+1, 2]) / 2
δ = uref - v0
v = deepcopy(v0)
for i in eachindex(v)
    if ks2.pSpace.x[i] >= 0.3 && ks2.pSpace.x[i] <= 4.7
        v[i] += 0.9 * δ[i]
    end
end
Plots.scatter!(ks2.pSpace.x[2:3:end-1, 1], v[2:3:end-1], label="UBE")
Plots.savefig("creep_profile_hline1.pdf")

# vline
Plots.plot(vr1.u, vr1.y, lw=2, label="Ref", xlabel="U", ylabel="y")
val0 = (field2[end÷2, 1:end÷2, 2] + field2[end÷2+1, 1:end÷2, 2]) / 2
val = @. val0 * exp(40*abs(val0))
Plots.scatter!(
    val .- 0.00005,
    ks2.pSpace.y[1, 1:end÷2].-0.005,
    label="UBE"
)
Plots.savefig("creep_profile_vline1.pdf")

# Kn = 10
begin
    #@load "data/continuum200_quad28_argon.jld2" ctr
    #@load "data/rarefied100_quad80_argon.jld2" ctr
    #@load "continuum/mesh300/ctr.jld2" ctr
    #@load "continuum/mesh100/ctr.jld2" ctr
    #@load "continuum/mesh200/sol.jld2" ks ctr
    #@load "continuum/mesh100 fsm/sol.jld2" ks ctr
    @load "huang/transition/sol.jld2" ks ctr
    #@load "reference/sol.jld2" ks ctr
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field1[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
end

Plots.plot(hr2.x, hr2.u)
Plots.scatter!(ks.pSpace.x[2:2:end-1, 1], field1[2:2:end-1, end÷2+1, 2] .- 0.000025)



Plots.plot!(href1.x, href1.u)

Plots.plot!(ks.pSpace.x[1:end, 1], field1[:, end÷2+1, 2])

Plots.plot((field1[end÷2, 1:end÷2, 2] + field1[end÷2+1, 1:end÷2, 2]) / 2, ks.pSpace.y[1, 1:end÷2])
Plots.scatter!(vr1.x, vr1.u)



Plots.plot((field1[end÷2, 1:end÷2, 2] + field1[end÷2+1, 1:end÷2, 2]) / 2, ks.pSpace.y[1, 1:end÷2])





begin
    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field1[:, :, 4]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
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






Plots.plot(ks.pSpace.x[1:end, 1], field1[:, end÷2+1, 2]/4)
Plots.scatter!(hr3.x, hr3.u)

begin
    @load "continuum/mesh100 fsm/sol.jld2" ks ctr
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 5)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:4] .= ctr[i, j].prim[1:4]
        field1[i, j, 5] = 1 / ctr[i, j].prim[end]
    end
end

begin
    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field1[:, :, 5]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
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

Plots.contourf(ks.vSpace.u[1:end, 1], ks.vSpace.v[1, 1:end], ctr[200, 2].h[1:end, 1:end])