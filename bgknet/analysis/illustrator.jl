using CairoMakie, KitBase, NipponColors
using KitBase.Distributions

dc = dict_color()
tc = plot_color()
cd(@__DIR__)

vs = VSpace1D(-5, 5, 200)
m = moment_basis(vs.u, 4)

#--- equilibrium ---#
e1 = begin
    prim = [1., -0.02, 1.]

    α = zeros(size(m, 1))
    α[1] = log((prim[1] / (π / prim[end]))^0.5) - prim[2]^2 * prim[end]
    α[2] = 2.0 * prim[2] * prim[end]
    α[3] = -prim[3]

    exp.(α' * m)[:]
end
e2 = begin
    prim = [1.0, 0.05, 1.]

    α = zeros(size(m, 1))
    α[1] = log((prim[1] / (π / prim[end]))^0.5) - prim[2]^2 * prim[end]
    α[2] = 2.0 * prim[2] * prim[end]
    α[3] = -prim[3]

    exp.(α' * m)[:]
end
e3 = @. e1 * heaviside(vs.u) + e2 * (1 - heaviside(vs.u))
e4 = f_maxwellian(e3, vs, 3)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, e1; color = dc["ruri"], label = "Left")
    lines!(vs.u, e2; color = dc["beniukon"], label = "Right")
    lines!(vs.u, e4; color = dc["ro"], label = "Maxwellian", linestyle = :dash)
    scatter!(vs.u, e3; color = (dc["tokiwa"], 0.7), label = "Reconstructed")
    axislegend()
    fig
end
save("recons_f1.pdf", fig)

#--- non-equilibrium ---#
f1 = begin
    prim = [1.5, 0.5, 0.5]

    α = zeros(size(m, 1))
    α[1] = log((prim[1] / (π / prim[end]))^0.5) - prim[2]^2 * prim[end]
    α[2] = 2.0 * prim[2] * prim[end]
    α[3] = -prim[3]

    α[4] = 0.1#1
    α[5] = -0.1#1

    exp.(α' * m)[:]
end
f2 = begin
    prim = [1., -0.5, 1.]

    α = zeros(size(m, 1))
    α[1] = log((prim[1] / (π / prim[end]))^0.5) - prim[2]^2 * prim[end]
    α[2] = 2.0 * prim[2] * prim[end]
    α[3] = -prim[3]

    α[4] = 0.005#1
    α[5] = 0#1

    exp.(α' * m)[:]
end
f3 = @. f1 * heaviside(vs.u) + f2 * (1 - heaviside(vs.u))
f4 = f_maxwellian(f3, vs, 3)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "u", ylabel = "f", title = "")
    lines!(vs.u, f1; color = dc["ruri"], label = "Left")
    lines!(vs.u, f2; color = dc["beniukon"], label = "Right")
    lines!(vs.u, f4; color = dc["ro"], label = "Maxwellian", linestyle = :dash)
    scatter!(vs.u, f3; color = (dc["tokiwa"], 0.7), label = "Reconstructed")
    axislegend()
    fig
end
save("recons_f2.pdf", fig)
