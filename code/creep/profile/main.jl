using Kinetic, PyPlot, DataFrames
using KitBase.ProgressMeter, KitBase.Plots, KitBase.JLD2
using KitML.CSV

cd(@__DIR__)
href1 = CSV.File("hline_kn0.08_argon.csv") |> DataFrame
href2 = CSV.File("hline_kn10.csv") |> DataFrame

function evolve(
    KS::SolverSet,
    ctr::T1,
    a1face::T2,
    a2face::T2,
    dt;
    mode = Symbol(KS.set.flux)::Symbol,
    bc = :fix::Symbol,
) where {T1<:AbstractArray{ControlVolume2D2F,2},T2<:AbstractArray{Interface2D2F,2}}

    if firstindex(KS.pSpace.x[:, 1]) < 1
        idx0 = 1
        idx1 = KS.pSpace.nx + 1
    else
        idx0 = 2
        idx1 = KS.pSpace.nx
    end
    if firstindex(KS.pSpace.y[1, :]) < 1
        idy0 = 1
        idy1 = KS.pSpace.ny + 1
    else
        idy0 = 2
        idy1 = KS.pSpace.ny
    end

    # x direction
    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        for i = idx0:idx1
            vn = KS.vSpace.u .* a1face[i, j].n[1] .+ KS.vSpace.v .* a1face[i, j].n[2]
            vt = KS.vSpace.v .* a1face[i, j].n[1] .- KS.vSpace.u .* a1face[i, j].n[2]

            flux_kfvs!(
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                ctr[i-1, j].h .+ 0.5 .* ctr[i-1, j].dx .* ctr[i-1, j].sh[:, :, 1],
                ctr[i-1, j].b .+ 0.5 .* ctr[i-1, j].dx .* ctr[i-1, j].sb[:, :, 1],
                ctr[i, j].h .- 0.5 .* ctr[i, j].dx .* ctr[i, j].sh[:, :, 1],
                ctr[i, j].b .- 0.5 .* ctr[i, j].dx .* ctr[i, j].sb[:, :, 1],
                vn,
                vt,
                KS.vSpace.weights,
                dt,
                a1face[i, j].len,
                ctr[i-1, j].sh[:, :, 1],
                ctr[i-1, j].sb[:, :, 1],
                ctr[i, j].sh[:, :, 1],
                ctr[i, j].sb[:, :, 1],
            )
            a1face[i, j].fw .=
                global_frame(a1face[i, j].fw, a1face[i, j].n[1], a1face[i, j].n[2])
        end
    end

    # y direction
    @inbounds Threads.@threads for j = idy0:idy1
        for i = 1:KS.pSpace.nx
            vn = KS.vSpace.u .* a2face[i, j].n[1] .+ KS.vSpace.v .* a2face[i, j].n[2]
            vt = KS.vSpace.v .* a2face[i, j].n[1] .- KS.vSpace.u .* a2face[i, j].n[2]

            flux_kfvs!(
                a2face[i, j].fw,
                a2face[i, j].fh,
                a2face[i, j].fb,
                ctr[i, j-1].h .+ 0.5 .* ctr[i, j-1].dy .* ctr[i, j-1].sh[:, :, 2],
                ctr[i, j-1].b .+ 0.5 .* ctr[i, j-1].dy .* ctr[i, j-1].sb[:, :, 2],
                ctr[i, j].h .- 0.5 .* ctr[i, j].dy .* ctr[i, j].sh[:, :, 2],
                ctr[i, j].b .- 0.5 .* ctr[i, j].dy .* ctr[i, j].sb[:, :, 2],
                vn,
                vt,
                KS.vSpace.weights,
                dt,
                a2face[i, j].len,
                ctr[i, j-1].sh[:, :, 2],
                ctr[i, j-1].sb[:, :, 2],
                ctr[i, j].sh[:, :, 2],
                ctr[i, j].sb[:, :, 2],
            )
            a2face[i, j].fw .=
                global_frame(a2face[i, j].fw, a2face[i, j].n[1], a2face[i, j].n[2])
        end
    end
    
    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        vn = KS.vSpace.u .* a1face[1, j].n[1] .+ KS.vSpace.v .* a1face[1, j].n[2]
        vt = KS.vSpace.v .* a1face[1, j].n[1] .- KS.vSpace.u .* a1face[1, j].n[2]
        bcL = local_frame(KS.ib.bcL, a1face[1, j].n[1], a1face[1, j].n[2])
        flux_boundary_maxwell!(
            a1face[1, j].fw,
            a1face[1, j].fh,
            a1face[1, j].fb,
            bcL, # left
            ctr[1, j].h,
            ctr[1, j].b,
            vn,
            vt,
            KS.vSpace.weights,
            KS.gas.K,
            dt,
            ctr[1, j].dy,
            1,
        )
        a1face[1, j].fw .=
            global_frame(a1face[1, j].fw, a1face[1, j].n[1], a1face[1, j].n[2])

        vn = KS.vSpace.u .* a1face[KS.pSpace.nx+1, j].n[1] .+ KS.vSpace.v .* a1face[KS.pSpace.nx+1, j].n[2]
        vt = KS.vSpace.v .* a1face[KS.pSpace.nx+1, j].n[1] .- KS.vSpace.u .* a1face[KS.pSpace.nx+1, j].n[2]
        bcR = local_frame(KS.ib.bcR, a1face[KS.pSpace.nx+1, j].n[1], a1face[KS.pSpace.nx+1, j].n[2])
        flux_boundary_maxwell!(
            a1face[KS.pSpace.nx+1, j].fw,
            a1face[KS.pSpace.nx+1, j].fh,
            a1face[KS.pSpace.nx+1, j].fb,
            bcR, # right
            ctr[KS.pSpace.nx, j].h,
            ctr[KS.pSpace.nx, j].b,
            vn,
            vt,
            KS.vSpace.weights,
            KS.gas.K,
            dt,
            ctr[KS.pSpace.nx, j].dy,
            -1,
        )
        a1face[KS.pSpace.nx+1, j].fw .= global_frame(
            a1face[KS.pSpace.nx+1, j].fw,
            a1face[KS.pSpace.nx+1, j].n[1],
            a1face[KS.pSpace.nx+1, j].n[2],
        )
    end

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        vn = KS.vSpace.u .* a2face[i, 1].n[1] .+ KS.vSpace.v .* a2face[i, 1].n[2]
        vt = KS.vSpace.v .* a2face[i, 1].n[1] .- KS.vSpace.u .* a2face[i, 1].n[2]

        _bcD = deepcopy(KS.ib.bcL)
        _T = 1/KS.ib.bcL[end] + (1/KS.ib.bcR[end] - 1/KS.ib.bcL[end]) * (KS.pSpace.x[i, 1] / 5.0)
        _bcD[end] = 1.0 / _T

        bcD = local_frame(_bcD, a2face[i, 1].n[1], a2face[i, 1].n[2])
        flux_boundary_maxwell!(
            a2face[i, 1].fw,
            a2face[i, 1].fh,
            a2face[i, 1].fb,
            bcD, # left
            ctr[i, 1].h,
            ctr[i, 1].b,
            vn,
            vt,
            KS.vSpace.weights,
            KS.gas.K,
            dt,
            ctr[i, 1].dx,
            1,
        )
        a2face[i, 1].fw .=
            global_frame(a2face[i, 1].fw, a2face[i, 1].n[1], a2face[i, 1].n[2])

        vn = KS.vSpace.u .* a2face[i, KS.pSpace.ny+1].n[1] .+ KS.vSpace.v .* a2face[i, KS.pSpace.ny+1].n[2]
        vt = KS.vSpace.v .* a2face[i, KS.pSpace.ny+1].n[1] .- KS.vSpace.u .* a2face[i, KS.pSpace.ny+1].n[2] 
        
        _bcU = deepcopy(KS.ib.bcL)
        _T = 1/KS.ib.bcL[end] + (1/KS.ib.bcR[end] - 1/KS.ib.bcL[end]) * (KS.pSpace.x[i, 1] / 5.0)
        _bcU[end] = 1.0 / _T

        bcU = local_frame(_bcU, a2face[i, KS.pSpace.ny+1].n[1], a2face[i, KS.pSpace.ny+1].n[2])
        flux_boundary_maxwell!(
            a2face[i, KS.pSpace.ny+1].fw,
            a2face[i, KS.pSpace.ny+1].fh,
            a2face[i, KS.pSpace.ny+1].fb,
            bcU, # right
            ctr[i, KS.pSpace.ny].h,
            ctr[i, KS.pSpace.ny].b,
            vn,
            vt,
            KS.vSpace.weights,
            KS.gas.K,
            dt,
            ctr[i, KS.pSpace.ny].dx,
            -1,
        )
        a2face[i, KS.pSpace.ny+1].fw .= global_frame(
            a2face[i, KS.pSpace.ny+1].fw,
            a2face[i, KS.pSpace.ny+1].n[1],
            a2face[i, KS.pSpace.ny+1].n[2],
        )
    end

end

cd(@__DIR__)

set = Setup(
    "gas", # matter
    "creep", # case
    "2d2f2v", # space
    "kfvs", # flux
    "bgk", # collision: for scalar conservation laws there are none
    1, # species
    2, # interpolation order
    "vanleer", # limiter
    "maxwell", # boundary
    0.8, # cfl
    100.0, # simulation time
)

#ps = PSpace2D(0.0, 5.0, 200, 0.0, 1.0, 40)
ps = PSpace2D(0.0, 5.0, 50, 0.0, 1.0, 10)
vs = VSpace2D(-5.0, 5.0, 72, -5.0, 5.0, 72, "rectangle")
Kn = 20.0
gas = KitBase.Gas(Kn, 0.0, 2/3, 1, 5/3, 0.81, 1.0, 0.5, ref_vhs_vis(Kn, 1.0, 0.5))

prim0 = [1.0, 0.0, 0.0, 1.0]
w0 = prim_conserve(prim0, 5/3)
h0 = maxwellian(vs.u, vs.v, prim0)
b0 = @. h0 * gas.K / (2 * prim0[end])
bcL = deepcopy(prim0)
bcR = [1.0, 0.0, 0.0, 0.5]
ib = IB2F(w0, prim0, h0, b0, bcL, w0, prim0, h0, b0, bcR)

ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)

ctr, a1face, a2face = init_fvm(ks, ks.pSpace)
#@load "ctr.jld2" ctr

for j in axes(ctr, 2), i in axes(ctr, 1)
    _T = 1/ks.ib.bcL[end] + (1/ks.ib.bcR[end] - 1/ks.ib.bcL[end]) * (ks.pSpace.x[i, 1] / 5.0)
    _λ = 1 / _T

    ctr[i, j].prim .= [_λ, 0.0, 0.0, _λ]
    ctr[i, j].w .= prim_conserve(ctr[i, j].prim, ks.gas.γ)
    ctr[i, j].h .= maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
    ctr[i, j].b .= ctr[i, j].h .* ks.gas.K ./ 2.0 ./ ctr[i, j].prim[end]
end
#=
@load "../finemesh/ctr.jld2" ctr
field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
for j in axes(field1, 2), i in axes(field1, 1)
    field1[i, j, :] .= ctr[i, j].prim
end
ctr, a1face, a2face = init_fvm(ks, ks.pSpace)
for j in axes(ctr, 2), i in axes(ctr, 1)
    ctr[i, j].prim .= field1[i, j, :]
    ctr[i, j].w .= prim_conserve(ctr[i, j].prim, ks.gas.γ)
    ctr[i, j].h .= maxwellian(ks.vSpace.u, ks.vSpace.v, ctr[i, j].prim)
    ctr[i, j].b .= ctr[i, j].h .* ks.gas.K ./ 2.0 ./ ctr[i, j].prim[end]
end=#

res = zeros(4)
t = 0.0
dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int

@showprogress for iter = 1:1000#nt
    reconstruct!(ks, ctr)
    evolve(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    #evolve!(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    Kinetic.update!(ks, ctr, a1face, a2face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))

    if maximum(res) < 1.e-7
        break
    end

    if iter%100 == 0
        println("loss: $(res)")
        @save "ctr.jld2" ctr
    end
end

#@save "ctr.jld2" ctr
begin
    close("all")
    field = zeros(ks.pSpace.nx, ks.pSpace.ny, 4)
    for j in axes(field, 2), i in axes(field, 1)
        field[i, j, 1:3] .= ctr[i, j].prim[1:3]
        field[i, j, 4] = 1 / ctr[i, j].prim[end]
    end
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field[:, :, 4]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    #colorbar()
    colorbar(orientation="horizontal")
    PyPlot.streamplot(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field[:, :, 2]', field[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
    xlabel("x")
    ylabel("y")
    #PyPlot.title("U-velocity")
    xlim(0.01,4.99)
    ylim(0.01,0.99)
    PyPlot.axes().set_aspect(1.2)
    #PyPlot.grid("on")
    display(fig)
    #fig.savefig("cavity_u.pdf")
end

#plot_contour(ks, ctr)
Plots.plot(ks.pSpace.x[1:end, 1], (field[:, end÷2, 2] .+ field[:, end÷2+1, 2])./2)