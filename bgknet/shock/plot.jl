using KitBase, CairoMakie, NipponColors
using KitBase.JLD2
using Flux: relu

dc = dict_color()
cd(@__DIR__)

function proc_sol(fname, ::Type{VDF{1,3}})
    @load fname ks ctr
    sol0 = extract_sol(ks, ctr)
    sol = zeros(ks.ps.nx, 8)
    for i in axes(sol, 1)
        sol[i, 1:5] .= sol0[i, :]
        sol[i, 6] = stress(ctr[i].f, ctr[i].prim, ks.vs.u, ks.vs.v, ks.vs.w, ks.vs.weights)[1, 1]
        sol[i, 7] = heat_flux(ctr[i].f, ctr[i].prim, ks.vs.u, ks.vs.v, ks.vs.w, ks.vs.weights)[1]
        sol[i, 8] = sum(ks.vs.weights .* ctr[i].f .* log.(relu(ctr[i].f) .+ eps()))
    end

    vs2 = VSpace2D(
        ks.vs.v0,
        ks.vs.v1,
        ks.vs.nv,
        ks.vs.w0,
        ks.vs.w1,
        ks.vs.nw,
    )
    solf = zeros(ks.ps.nx, ks.vs.nu)
    for i in axes(solf, 1)
        solf[i, :] .= reduce_distribution(ctr[i].f, vs2.weights)
    end

    col = zeros(ks.ps.nx, ks.vs.nu)
    _col = zeros(ks.vs.nu, ks.vs.nv, ks.vs.nw)
    if ks.gas.fsm isa Nothing
        for i in axes(col, 1)
            M = maxwellian(ks.vs.u, ks.vs.v, ks.vs.w, ctr[i].prim)
            τ = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
            @. _col = (M - ctr[i].f) / τ
            col[i, :] .= reduce_distribution(_col, vs2.weights)
        end
    else
        for i in axes(col, 1)
            _col .= boltzmann_fft(ctr[i].f, ks.gas.fsm.Kn, ks.gas.fsm.nm, ks.gas.fsm.ϕ, ks.gas.fsm.ψ, ks.gas.fsm.χ)
            col[i, :] .= reduce_distribution(_col, vs2.weights)
        end
    end

    return ks, sol, solf, col
end

function proc_sol(fname, ::Type{VDF{2,1}})
    @load fname ks ctr
    sol0 = extract_sol(ks, ctr)
    sol1 = zeros(ks.ps.nx, 6)
    for i in axes(sol1, 1)
        sol1[i, 1:3] .= sol0[i, :]
        sol1[i, 4] = stress(ctr[i].h, ctr[i].prim, ks.vs.u, ks.vs.weights)
        sol1[i, 5] = heat_flux(ctr[i].h, ctr[i].b, ctr[i].prim, ks.vs.u, ks.vs.weights)
        sol1[i, 6] = sum(ks.vs.weights .* ctr[i].h .* log.(relu(ctr[i].h) .+ eps()))
    end

    solf1 = zeros(ks.ps.nx, ks.vs.nu)
    for i in axes(solf1, 1)
        solf1[i, :] .= ctr[i].h
    end

    return ks, sol1, solf1
end

ks, sol, solf, col = proc_sol("sol3d_ma2.jld2", VDF{1,3})
ks1, sol1, solf1, col1 = proc_sol("sol3d_ma2_bgk.jld2", VDF{1,3})
ks11, sol11, solf11 = proc_sol("sol1d_ma2.jld2", VDF{2,1})

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks.ps.x[1:ks.ps.nx], ks.vs.u[:, 1, 1], solf; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_contour_ube_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks1.ps.x[1:ks.ps.nx], ks1.vs.u[:, 1, 1], solf1; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_contour_bgk_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks.ps.x[1:ks.ps.nx], ks.vs.u[:, 1, 1], col; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_collision_ube_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks.ps.x[1:ks.ps.nx], ks.vs.u[:, 1, 1], col1; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_collision_bgk_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 1]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 1]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 1]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_density_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "U", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 2]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 2]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 2]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_velocity_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "T", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, 5]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], 1 ./ sol1[:, 5]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, 5]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_temperature_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "ϖ", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 6] .- 0.5 .* sol[:, 1] ./ sol[:, 5]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 6] .- 0.5 .* sol1[:, 1] ./ sol1[:, 5]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 6] .- 0.5 .* sol[:, 1] ./ sol[:, 5]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_stress_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "q", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 7]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 7]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 7]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_heat_ma2.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "Entropy", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 8]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 8]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 8]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_entropy_ma2.pdf", fig)

"""
Ma = 3
"""

ks, sol, solf, col = proc_sol("sol3d_ma3.jld2", VDF{1,3})
ks1, sol1, solf1, col1 = proc_sol("sol3d_ma3_bgk.jld2", VDF{1,3})
ks11, sol11, solf11 = proc_sol("sol1d_ma3.jld2", VDF{2,1})

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks.ps.x[1:ks.ps.nx], ks.vs.u[:, 1, 1], solf; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_contour_ube_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks1.ps.x[1:ks.ps.nx], ks1.vs.u[:, 1, 1], solf1; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_contour_bgk_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks.ps.x[1:ks.ps.nx], ks.vs.u[:, 1, 1], col; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_collision_ube_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", title = "")
    co = contourf!(ks.ps.x[1:ks.ps.nx], ks.vs.u[:, 1, 1], col1; colormap = :PiYG_8, levels = 15)
    Colorbar(fig[1, 2], co)
    fig
end
save("shock_collision_bgk_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 1]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 1]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 1]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_density_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "U", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 2]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 2]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 2]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_velocity_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "T", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, 5]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], 1 ./ sol1[:, 5]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], 1 ./ sol[:, 5]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_temperature_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "ϖ", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 6] .- 0.5 .* sol[:, 1] ./ sol[:, 5]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 6] .- 0.5 .* sol1[:, 1] ./ sol1[:, 5]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 6] .- 0.5 .* sol[:, 1] ./ sol[:, 5]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_stress_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "q", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 7]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 7]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 7]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_heat_ma3.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "Entropy", title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 8]; color = dc["ro"], label = "Boltzmann")
    lines!(ks.ps.x[1:ks.ps.nx], sol1[:, 8]; color = dc["ruri"], label = "BGK", linestyle = :dash)
    scatter!(ks.ps.x[1:ks.ps.nx], sol[:, 8]; color = (dc["tokiwa"], 0.7), label = "UBE")
    axislegend()
    fig
end
save("shock_entropy_ma3.pdf", fig)
