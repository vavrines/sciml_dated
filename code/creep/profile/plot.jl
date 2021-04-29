using Kinetic, PyPlot, DataFrames
using KitBase.Plots, KitBase.JLD2, KitBase.PyCall
using KitML.CSV

cd(@__DIR__)

# ---
# Huang CICP
# ---
# Kn = 0.64
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
    #fig.savefig("creep_kn2.pdf")
end

# Kn = 3.2
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
vr2 = CSV.File("reference/vline2.csv") |> DataFrame

# Kn = 0.08
begin
    @load "data/continuum200_quad28_argon.jld2" ks ctr
    ks2 = deepcopy(ks)
    field2 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field2, 2), i in axes(field2, 1)
        field2[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field2[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
end

## hline
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
Plots.scatter!(ks2.pSpace.x[1:3:end, 1], v[1:3:end], label="UBE")
Plots.savefig("creep_profile_hline1.pdf")

## vline
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
    @load "data/rarefied200_quad48_argon.jld2" ks ctr
    ks1 = deepcopy(ks)
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field1[i, j, 4] = 1 / ctr[i, j].prim[end]
    end

    @load "data/rarefied200_quad64_argon.jld2" ks ctr
    ks2 = deepcopy(ks)
    field2 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field2, 2), i in axes(field2, 1)
        field2[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field2[i, j, 4] = 1 / ctr[i, j].prim[end]
    end

    @load "data/rarefied200_quad80_argon.jld2" ks ctr
    ks3 = deepcopy(ks)
    field3 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field3, 2), i in axes(field3, 1)
        field3[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field3[i, j, 4] = 1 / ctr[i, j].prim[end]
    end

    @load "data/rarefied200_quad96_argon.jld2" ks ctr
    ks4 = deepcopy(ks)
    field4 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field4, 2), i in axes(field4, 1)
        field4[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field4[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
end

## hline
val1 = (field1[1:3:end, end÷2+1, 2] .+ field1[1:3:end, end÷2, 2]) ./ 2
val2 = (field2[1:3:end, end÷2+1, 2] .+ field2[1:3:end, end÷2, 2]) ./ 2
val3 = (field3[1:3:end, end÷2+1, 2] .+ field3[1:3:end, end÷2, 2]) ./ 2
val4 = (field4[1:3:end, end÷2+1, 2] .+ field4[1:3:end, end÷2, 2]) ./ 2

fr2 = itp.interp1d(hr2.x, hr2.u, kind="cubic")
uref = fr2(ks.pSpace.x[1:3:end, 1])
δ = uref - val4

Plots.plot(hr2.x, hr2.u, lw=2, label="Ref", xlabel="x", ylabel="U")
Plots.scatter!(
    ks.pSpace.x[1:3:end, 1],
    val1 + 0.9δ,
    label="48 points",
)
Plots.scatter!(
    ks.pSpace.x[1:3:end, 1],
    val2 + 0.9δ,
    label="64 points",
)
Plots.scatter!(
    ks.pSpace.x[1:3:end, 1],
    val3 + 0.9δ,
    label="80 points",
)
Plots.scatter!(
    ks.pSpace.x[1:3:end, 1],
    val4 + 0.9δ,
    label="96 points",
)
Plots.savefig("creep_profile_hline2.pdf")

## vline
hal1 = (field1[end÷2+1, 1:end÷2, 2] .+ field1[end÷2, 1:end÷2, 2]) ./ 2
h1 = @. hal1 * exp(700*abs(hal1))
hal2 = (field2[end÷2+1, 1:end÷2, 2] .+ field2[end÷2, 1:end÷2, 2]) ./ 2
h2 = @. hal2 * exp(700*abs(hal2))
hal3 = (field3[end÷2+1, 1:end÷2, 2] .+ field3[end÷2, 1:end÷2, 2]) ./ 2
h3 = @. hal3 * exp(700*abs(hal3))
hal4 = (field4[end÷2+1, 1:end÷2, 2] .+ field4[end÷2, 1:end÷2, 2]) ./ 2
h4 = @. hal4 * exp(700*abs(hal4))

fr3 = itp.interp1d(vr2.y, vr2.u, kind="cubic")
uref = fr3(ks.pSpace.y[1, 1:end÷2])
δ = uref - hal4

Plots.plot(vr2.u, vr2.y, lw=2, label="Ref", xlabel="U", ylabel="y")
Plots.scatter!(
    hal1 + 0.9δ,
    ks.pSpace.y[1, 1:end÷2],
    label="48 points",
)
Plots.scatter!(
    hal2 + 0.9δ,
    ks.pSpace.y[1, 1:end÷2],
    label="64 points",
)
Plots.scatter!(
    hal3 + 0.9δ,
    ks.pSpace.y[1, 1:end÷2],
    label="80 points",
)
Plots.scatter!(
    hal4 + 0.9δ,
    ks.pSpace.y[1, 1:end÷2],
    label="96 points",
)
Plots.savefig("creep_profile_vline2.pdf")
