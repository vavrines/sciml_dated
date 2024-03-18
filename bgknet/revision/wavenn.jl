using Kinetic, Plots
using KitBase.JLD2
using Flux: elu
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

function update_ube!(
	KS,
	ctr,
	face,
	dt,
	residual,
	nn,
	p,
)

	sumRes = zero(ctr[1].w)
	sumAvg = zero(ctr[1].w)
	nm = param_length(nn.Mnet)

	@inbounds @threads for i ∈ 1:KS.ps.nx
		step_ube!(
			ctr[i].w,
			ctr[i].prim,
			ctr[i].f,
			face[i].fw,
			face[i].ff,
			face[i+1].fw,
			face[i+1].ff,
			vs.u,
			ks.gas.γ,
			ks.gas.μᵣ,
			ks.gas.ω,
			ks.ps.dx[i],
			dt,
			sumRes,
			sumAvg,
			nn,
			p,
			nm,
		)
	end

	for i in eachindex(residual)
		residual[i] = sqrt(sumRes[i] * KS.ps.nx) / (sumAvg[i] + 1.e-7)
	end

	ctr[0].w .= ctr[ks.ps.nx].w
	ctr[0].sw .= ctr[ks.ps.nx].sw
	ctr[0].prim .= ctr[ks.ps.nx].prim
	ctr[0].f .= ctr[ks.ps.nx].f
	ctr[0].sf .= ctr[ks.ps.nx].sf
	ctr[ks.ps.nx+1].w .= ctr[1].w
	ctr[ks.ps.nx+1].sw .= ctr[1].sw
	ctr[ks.ps.nx+1].prim .= ctr[1].prim
	ctr[ks.ps.nx+1].f .= ctr[1].f
	ctr[ks.ps.nx+1].sf .= ctr[1].sf

	return nothing

end

function step_ube!(
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
	nn,
	p,
	nm,
) where {T1, T2, T3, T4, T5}
	w0 = deepcopy(w)
	prim0 = conserve_prim(w0, γ)
	τ0 = vhs_collision_time(prim0, μᵣ, ω)
	M0 = maxwellian(u, prim0)

	y = f .- M0
	α = nn.Mnet(y, p[1:nm])
	S = collision_invariant(α, u)

	z = vcat(y, τ0)
	τ = τ0 .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end])))

	#--- update W^{n+1} ---#
	@. w += (fwL - fwR) / dx
	prim .= conserve_prim(w, γ)

	#--- record residuals ---#
	@. RES += (w - w0)^2
	@. AVG += abs(w)

	#--- update distribution function ---#
	for i in eachindex(u)
		f[i] = f[i] + (ffL[i] - ffR[i]) / dx + dt / τ[i] * (M0[i] * S[i] - f[i])
	end
end

cd(@__DIR__)
@load "../relaxation/shakhov/prototype.jld2" nn u

cf = config_ntuple(
	t1 = 1.0,
	x0 = 0,
	x1 = 1,
	nx = 200,
	u0 = -8,
	u1 = 8,
	nu = 80,
	K = 0,
	Kn = 0.001,
)

set = Setup(case = "sod", space = "1d1f1v", collision = "bgk", maxTime = cf.t1)
ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = VSpace1D(cf.u0, cf.u1, cf.nu)
gas = Gas(Kn = cf.Kn, K = cf.K, γ = 3)

p = (u = vs.u,)
fw = function (x, p)
	ρ = 1.0 + 0.1 * sin(2π * x)
	u = 1.0
	t = 2 / ρ

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

ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)

dt = 0.0002#timestep(ks, ctr, 0)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(3)

@showprogress for iter ∈ 1:nt÷4
	reconstruct!(ks, ctr)
	evolve!(ks, ctr, face, dt; bc = :period)
	#update!(ks, ctr, face, dt, res; bc = :period)
	update_ube!(ks, ctr, face, dt, res, nn, u)
end

Plots.plot(ks, ctr)
