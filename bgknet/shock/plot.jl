using KitBase, CairoMakie, NipponColors
using KitBase.JLD2

dc = dict_color()
cd(@__DIR__)

@load "sol3d.jld2" ks ctr
sol = extract_sol(ks, ctr)

@load "sol1d.jld2" ks ctr
sol1 = extract_sol(ks, ctr)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 1]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 1]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    axislegend()
    fig
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "U", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 2]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 2]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    axislegend()
    fig
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "T", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, end]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], 1 ./ sol1[:, end]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    axislegend()
    fig
end
