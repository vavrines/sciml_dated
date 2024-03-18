using Kinetic, Plots
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)

cf = config_ntuple(
	t1 = 1.0,
	x0 = 0,
	x1 = 1,
	nx = 800,
	u0 = -8,
	u1 = 8,
	nu = 100,
	K = 0,
	Kn = 0.0001,
)

set = Setup(case = "sod", space = "1d1f1v", collision = "bgk", maxTime = cf.t1, limiter = "linear")
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

dt = 2e-5#timestep(ks, ctr, 0)
nt = ks.set.maxTime / dt |> ceil |> Int
res = zeros(3)

@showprogress for iter ∈ 1:nt
	reconstruct!(ks, ctr)
	evolve!(ks, ctr, face, dt; bc = :period)
	update!(ks, ctr, face, dt, res; bc = :period)
end

@save "wavref.jld2" ctr
