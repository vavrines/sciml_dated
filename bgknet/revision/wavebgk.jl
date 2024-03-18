using Kinetic, Plots
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

function fluxt!(face, ctrL, ctrR, vs, p, dt)
    dxL, dxR = p[1:2]

    flux_kfvs!(
        face.fw,
        face.ff,
        ctrL.f + ctrL.sf * dxL,
        ctrR.f - ctrR.sf * dxR,
        vs.u,
        vs.weights,
        dt,
        zero(ctrL.f),
        zero(ctrR.f),
    )

    return nothing
end

function stept!(
    w::T3,
    prim::T3,
    f::T4,
    fwL::T1,
    ffL::T2,
    fwR::T1,
    ffR::T2,
    u::T5,
    γ,
    μᵣ,
    ω,
    dx,
    dt,
    RES,
    AVG,
) where {T1,T2,T3,T4,T5}

    #--- store W^n and calculate H^n,\tau^n ---#
    w_old = deepcopy(w)
	M_old = maxwellian(u, prim)
	τ_old = vhs_collision_time(prim, μᵣ, ω)

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR) / dx
    prim .= conserve_prim(w, γ)

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    #--- calculate M^{n+1} and tau^{n+1} ---#
    M = maxwellian(u, prim)
    τ = vhs_collision_time(prim, μᵣ, ω)

    #--- update distribution function ---#
    for i in eachindex(u)
        f[i] = (f[i] + (ffL[i] - ffR[i]) / dx + 0.5 * dt * (M[i]/τ+(M_old[i]-f[i])/τ_old)) / (1.0 + 0.5 * dt / τ)
    end
end

cd(@__DIR__)
begin
	@load "wavref_0.1.jld2" ctr
	ctr0 = deepcopy(ctr)
	ps0 = PSpace1D(0, 1, 800, 1)
end

cf = config_ntuple(
    t1 = 1.0,
    x0 = 0,
    x1 = 1,
    nx = 5,
    u0 = -5,
    u1 = 5,
    nu = 200,
    K = 0,
    Kn = 0.1,
)

begin
	set = Setup(
		case = "sod",
		space = "1d1f1v",
		collision = "bgk",
		maxTime = cf.t1,
		limiter = "linear",
	)
	ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
	vs = VSpace1D(cf.u0, cf.u1, cf.nu)
	gas = Gas(Kn = cf.Kn, K = cf.K, γ = 3)

	p = (u = vs.u,)
	fw = function (x, p)
		ρ = 1.0 + 0.1 * sin(2π * x)
		u = 1.0
		t = 1.0 / ρ

		return prim_conserve([ρ, u, 1 / t], 3)
	end
	ff = function (x, p)
		w = fw(x, p)
		prim = conserve_prim(w, 3)
		return maxwellian(p.u, prim)
	end
	bc = function (x, p)
		w = fw(x, p)
		return conserve_prim(w, 3)
	end

	ib = IB1F(fw, ff, bc, p)
end

ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)

dt = 2e-5#timestep(ks, ctr, 0)
nt = ks.set.maxTime / dt |> ceil |> Int
res = zeros(3)

@showprogress for iter ∈ 1:nt
    reconstruct!(ks, ctr)

	#evolve!(ks, ctr, face, dt; bc = :period)
    @threads for i = 1:ks.ps.nx+1
        @inbounds fluxt!(
            face[i],
            ctr[i-1],
            ctr[i],
            ks.vs,
            (0.5 * ks.ps.dx[i-1], 0.5 * ks.ps.dx[i]),
            dt,
        )
    end

    #update!(ks, ctr, face, dt, res; bc = :period)
	@threads for i = 1:ks.ps.nx
        @inbounds stept!(
			ctr[i].w,
			ctr[i].prim,
			ctr[i].f,
			face[i].fw,
			face[i].ff,
			face[i+1].fw,
			face[i+1].ff,
			ks.vs.u,
			ks.gas.γ,
			ks.gas.μᵣ,
			ks.gas.ω,
			ks.ps.dx[i],
			dt,
			zeros(3),
			zeros(3),
		)
    end
	KB.bc_period!(ctr)
end
#@save "wavref.jld2" ctr

begin
	sol = extract_sol(ps, ctr)
	sol0 = extract_sol(ps0, ctr0)

	fitp = KB.itp.interp1d(ps0.x[1:ps0.nx], sol0[:, 1], kind = "cubic")
	x, dx = ps.x[1:ps.nx], ps.dx[1]
    
	KB.L1_error(sol[:, 1], fitp(x), dx) |> println
    KB.L2_error(sol[:, 1], fitp(x), dx) |> println
    KB.L∞_error(sol[:, 1], fitp(x), dx) |> println
end

"""
dx L1 error L2 error L∞ error
Kn=0.001
0.2 0.05133904100807753 0.025638754858128786 0.015701698270736707
0.1 0.026926630311768864 0.009571965796080962 0.004665933887488527
0.05 0.011318240618447273 0.0028795840147793086 0.0011032344220005986
0.025 0.00478468090859629 0.000859237450457056 0.0002494705344461057

Kn=0.01
0.2 0.043436719438984174 0.021580421216029773 0.01298299845764437
0.1 0.022654175865524612 0.008075906295004273 0.003962220558214247
0.05 0.009552014504268904 0.002455254794356335 0.0009315556591882213
0.025 0.0040723670772496055 0.0007390609903701791 0.0002113526263258725

Kn=0.1
0.2 0.013619929373585693 0.006833368137666353 0.0044722758968051
0.1 0.0073241071268481 0.0025318941077998418 0.0011208425130641464
0.05 0.0030728818183303704 0.0007838322030583129 0.0002741623447861552
0.025 0.001301635390563477 0.00023872544223861185 6.550283350795772e-5
"""

KB.convergence_order(0.01298299845764437, 0.003962220558214247)

KB.convergence_order(0.003962220558214247,0.0009315556591882213)


plot(ks, ctr)

fo = KB.convergence_order

fo(1.29830E-2,2.315906295004E-3)
