using Kinetic, PyPlot, DataFrames
using KitBase.ProgressMeter, KitBase.Plots, KitBase.JLD2
using KitML.CSV

function evolve(
    KS::SolverSet,
    ctr::T1,
    a1face::T2,
    a2face::T2,
    dt;
    mode = Symbol(KS.set.flux)::Symbol,
    bc = :fix::Symbol,
) where {T1<:AbstractArray{ControlVolume2D1F,2},T2<:AbstractArray{Interface2D1F,2}}

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
                a1face[i, j].ff,
                ctr[i-1, j].f .+ 0.5 .* ctr[i-1, j].dx .* ctr[i-1, j].sf[:, :, :, 1],
                ctr[i, j].f .- 0.5 .* ctr[i, j].dx .* ctr[i, j].sf[:, :, :, 1],
                vn,
                vt,
                KS.vSpace.w,
                KS.vSpace.weights,
                dt,
                a1face[i, j].len,
                ctr[i-1, j].sf[:, :, :, 1],
                ctr[i, j].sf[:, :, :, 1],
            )
            a1face[i, j].fw[2:3] .=
                global_frame(a1face[i, j].fw[2:3], a1face[i, j].n[1], a1face[i, j].n[2])
        end
    end

    # y direction
    @inbounds Threads.@threads for j = idy0:idy1
        for i = 1:KS.pSpace.nx
            vn = KS.vSpace.u .* a2face[i, j].n[1] .+ KS.vSpace.v .* a2face[i, j].n[2]
            vt = KS.vSpace.v .* a2face[i, j].n[1] .- KS.vSpace.u .* a2face[i, j].n[2]

            flux_kfvs!(
                a2face[i, j].fw,
                a2face[i, j].ff,
                ctr[i, j-1].f .+ 0.5 .* ctr[i, j-1].dy .* ctr[i, j-1].sf[:, :, :, 2],
                ctr[i, j].f .- 0.5 .* ctr[i, j].dy .* ctr[i, j].sf[:, :, :, 2],
                vn,
                vt,
                KS.vSpace.w,
                KS.vSpace.weights,
                dt,
                a2face[i, j].len,
                ctr[i, j-1].sf[:, :, :, 2],
                ctr[i, j].sf[:, :, :, 2],
            )
            a2face[i, j].fw[2:3] .=
                global_frame(a2face[i, j].fw[2:3], a2face[i, j].n[1], a2face[i, j].n[2])
        end
    end
    
    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        vn = KS.vSpace.u .* a1face[1, j].n[1] .+ KS.vSpace.v .* a1face[1, j].n[2]
        vt = KS.vSpace.v .* a1face[1, j].n[1] .- KS.vSpace.u .* a1face[1, j].n[2]
        bcL = deepcopy(KS.ib.bcL)
        bcL[2:3] .= local_frame(KS.ib.bcL[2:3], a1face[1, j].n[1], a1face[1, j].n[2])
        flux_boundary_maxwell!(
            a1face[1, j].fw,
            a1face[1, j].ff,
            bcL, # left
            ctr[1, j].f .- 0.5 .* ctr[1, j].dx .* ctr[1, j].sf[:, :, :, 1],
            vn,
            vt,
            KS.vSpace.w,
            KS.vSpace.weights,
            dt,
            ctr[1, j].dy,
            1,
        )
        a1face[1, j].fw[2:3] .=
            global_frame(a1face[1, j].fw[2:3], a1face[1, j].n[1], a1face[1, j].n[2])

        vn = KS.vSpace.u .* a1face[KS.pSpace.nx+1, j].n[1] .+ KS.vSpace.v .* a1face[KS.pSpace.nx+1, j].n[2]
        vt = KS.vSpace.v .* a1face[KS.pSpace.nx+1, j].n[1] .- KS.vSpace.u .* a1face[KS.pSpace.nx+1, j].n[2]
        bcR = deepcopy(KS.ib.bcR)
        bcR[2:3] .= local_frame(KS.ib.bcR[2:3], a1face[KS.pSpace.nx+1, j].n[1], a1face[KS.pSpace.nx+1, j].n[2])
        flux_boundary_maxwell!(
            a1face[KS.pSpace.nx+1, j].fw,
            a1face[KS.pSpace.nx+1, j].ff,
            bcR, # right
            ctr[KS.pSpace.nx, j].f .+ 0.5 .* ctr[KS.pSpace.nx, j].dx .* ctr[KS.pSpace.nx, j].sf[:, :, :, 1],
            vn,
            vt,
            KS.vSpace.w,
            KS.vSpace.weights,
            dt,
            ctr[KS.pSpace.nx, j].dy,
            -1,
        )
        a1face[KS.pSpace.nx+1, j].fw[2:3] .= global_frame(
            a1face[KS.pSpace.nx+1, j].fw[2:3],
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

        bcD = deepcopy(_bcD)
        bcD[2:3] .= local_frame(_bcD[2:3], a2face[i, 1].n[1], a2face[i, 1].n[2])
        flux_boundary_maxwell!(
            a2face[i, 1].fw,
            a2face[i, 1].ff,
            bcD, # left
            ctr[i, 1].f .- 0.5 .* ctr[i, 1].dy .* ctr[i, 1].sf[:, :, :, 2],
            vn,
            vt,
            KS.vSpace.w,
            KS.vSpace.weights,
            dt,
            ctr[i, 1].dx,
            1,
        )
        a2face[i, 1].fw[2:3] .=
            global_frame(a2face[i, 1].fw[2:3], a2face[i, 1].n[1], a2face[i, 1].n[2])

        vn = KS.vSpace.u .* a2face[i, KS.pSpace.ny+1].n[1] .+ KS.vSpace.v .* a2face[i, KS.pSpace.ny+1].n[2]
        vt = KS.vSpace.v .* a2face[i, KS.pSpace.ny+1].n[1] .- KS.vSpace.u .* a2face[i, KS.pSpace.ny+1].n[2] 
        
        _bcU = deepcopy(KS.ib.bcL)
        _T = 1/KS.ib.bcL[end] + (1/KS.ib.bcR[end] - 1/KS.ib.bcL[end]) * (KS.pSpace.x[i, 1] / 5.0)
        _bcU[end] = 1.0 / _T

        bcU = deepcopy(_bcU)
        bcU[2:3] .= local_frame(_bcU[2:3], a2face[i, KS.pSpace.ny+1].n[1], a2face[i, KS.pSpace.ny+1].n[2])
        flux_boundary_maxwell!(
            a2face[i, KS.pSpace.ny+1].fw,
            a2face[i, KS.pSpace.ny+1].ff,
            bcU, # right
            ctr[i, KS.pSpace.ny].f .+ 0.5 .* ctr[i, KS.pSpace.ny].dy .* ctr[i, KS.pSpace.ny].sf[:, :, :, 2],
            vn,
            vt,
            KS.vSpace.w,
            KS.vSpace.weights,
            dt,
            ctr[i, KS.pSpace.ny].dx,
            -1,
        )
        a2face[i, KS.pSpace.ny+1].fw[2:3] .= global_frame(
            a2face[i, KS.pSpace.ny+1].fw[2:3],
            a2face[i, KS.pSpace.ny+1].n[1],
            a2face[i, KS.pSpace.ny+1].n[2],
        )
    end

end

function step(
    w::T3,
    prim::T3,
    f::T4,
    fwL::T1,
    ffL::T2,
    fwR::T1,
    ffR::T2,
    fwD::T1,
    ffD::T2,
    fwU::T1,
    ffU::T2,
    γ,
    Kn_bz,
    nm,
    phi,
    psi,
    phipsi,
    Δs,
    dt,
    RES,
    AVG,
    collision = :fsm::Symbol,
) where {
    T1<:AbstractArray{<:AbstractFloat,1},
    T2<:AbstractArray{<:AbstractFloat,3},
    T3<:AbstractArray{<:AbstractFloat,1},
    T4<:AbstractArray{<:AbstractFloat,3},
}

    @assert collision == :fsm

    w_old = deepcopy(w)
    @. w += (fwL - fwR + fwD - fwU) / Δs

    prim .= conserve_prim(w, γ)

    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    Q = zero(f[:, :, :])
    #boltzmann_fft!(Q, f, Kn_bz, nm, phi, psi, phipsi)

    for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
        f[i, j, k] += (ffL[i, j, k] - ffR[i, j, k] + ffD[i, j, k] - ffU[i, j, k]) / Δs + dt * Q[i, j, k]
    end

end

function update(
    KS::X,
    ctr::Y,
    a1face::Z,
    a2face::Z,
    dt,
    residual,
    Kn_bz,
    nm,
    phi,
    psi,
    phipsi,
) where {
    X<:AbstractSolverSet,
    Y<:AbstractArray{ControlVolume2D1F,2},
    Z<:AbstractArray{Interface2D1F,2},
}

    sumRes = zero(KS.ib.wL)
    sumAvg = zero(KS.ib.wL)

    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx
            KitBase.step!(
            #step(
                ctr[i, j].w,
                ctr[i, j].prim,
                ctr[i, j].f,
                a1face[i, j].fw,
                a1face[i, j].ff,
                a1face[i+1, j].fw,
                a1face[i+1, j].ff,
                a2face[i, j].fw,
                a2face[i, j].ff,
                a2face[i, j+1].fw,
                a2face[i, j+1].ff,
                KS.gas.γ,
                Kn_bz,
                nm,
                phi,
                psi,
                phipsi,
                ctr[i, j].dx * ctr[i, j].dy,
                dt,
                sumRes,
                sumAvg,
                :fsm,
            )
        end
    end

    for i in eachindex(residual)
        residual[i] = sqrt(sumRes[i] * KS.pSpace.nx * KS.pSpace.ny) / (sumAvg[i] + 1.e-7)
    end

    return nothing

end

cd(@__DIR__)

set = Setup(
    "gas", # matter
    "creep", # case
    "2d1f3v", # space
    "kfvs", # flux
    "fsm", # collision: for scalar conservation laws there are none
    1, # species
    2, # interpolation order
    "vanleer", # limiter
    "maxwell", # boundary
    0.8, # cfl
    100.0, # simulation time
)

ps = PSpace2D(0.0, 5.0, 100, 0.0, 1.0, 20)
vs = VSpace3D(-7.5, 7.5, 64, -7.5, 7.5, 64, -7.5, 7.5, 16, "rectangle")
Kn = 10.0
gas = KitBase.Gas(Kn, 0.0, 2/3, 1, 5/3, 0.5, 1.0, 0.5, ref_vhs_vis(Kn, 1.0, 0.5))

prim0 = [1.0, 0.0, 0.0, 0.0, 1.0]
w0 = prim_conserve(prim0, 5 / 3)
f0 = maxwellian(vs.u, vs.v, vs.w, prim0)
bcL = deepcopy(prim0)
bcR = [1.0, 0.0, 0.0, 0.0, 0.5]
ib = IB1F(w0, prim0, f0, bcL, w0, prim0, f0, bcR)

ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)

ctr, a1face, a2face = init_fvm(ks, ks.pSpace)
for j in axes(ctr, 2), i in axes(ctr, 1)
    ctr[i, j].prim .= [1.0, 0.0, 0.0, 0.0, 1.0]
    ctr[i, j].w .= prim_conserve(ctr[i, j].prim, ks.gas.γ)
    ctr[i, j].f .= maxwellian(ks.vSpace.u, ks.vSpace.v, ks.vSpace.w, ctr[i, j].prim)
end

res = zeros(5)
t = 0.0
dt = timestep(ks, ctr, t)
nt = floor(ks.set.maxTime / dt) |> Int

nm = 5
phi, psi, phipsi = kernel_mode( nm, ks.vSpace.u1, ks.vSpace.v1, ks.vSpace.w1, ks.vSpace.du[1,1,1], ks.vSpace.dv[1,1,1], ks.vSpace.dw[1,1,1],
    ks.vSpace.nu, ks.vSpace.nv, ks.vSpace.nw, 1.0 )
kn_bzm = hs_boltz_kn(ks.gas.μᵣ, 1.0)

@showprogress for iter = 1:1#nt
    reconstruct!(ks, ctr)
    evolve(ks, ctr, a1face, a2face, dt; mode = Symbol(ks.set.flux), bc = Symbol(ks.set.boundary))
    update(ks, ctr, a1face, a2face, dt, res, kn_bzm, nm, phi, psi, phipsi)

    if maximum(res) < 1.e-7
        break
    end

    if iter%100 == 0
        println("loss: $(res)")
        @save "sol.jld2" ks ctr
    end
end

begin
    field1 = zeros(ks.pSpace.nx, ks.pSpace.ny, 5)
    for j in axes(field1, 2), i in axes(field1, 1)
        field1[i, j, 1:4] .= ctr[i, j].prim[1:4]
        field1[i, j, 5] = 1 / ctr[i, j].prim[end]
    end

    close("all")
    fig = figure("contour", figsize=(6.5, 5))
    PyPlot.contourf(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field1[:, :, 5]', linewidth=1, levels=20, cmap=ColorMap("inferno"))
    #colorbar()
    colorbar(orientation="horizontal")
    #PyPlot.streamplot(ks.pSpace.x[1:end, 1], ks.pSpace.y[1, 1:end], field1[:, :, 2]', field1[:, :, 3]', density=1.3, color="moccasin", linewidth=1)
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
