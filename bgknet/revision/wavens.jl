using Kinetic, Plots
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

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

set = Setup(case = "wave", space = "1d0f0v", maxTime = cf.t1, limiter = "linear", flux = "hll")
ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = nothing
gas = Gas(Kn = cf.Kn, K = cf.K, γ = 3)

fw = function (x, p)
	ρ = 1.0 + 0.1 * sin(2π * x)
	u = 1.0
	t = 2 / ρ

	return prim_conserve([ρ, u, 1 / t], 3)
end
bc = function (x, p)
	w = fw(x, p)
	return conserve_prim(w, 3)
end

ib = IB(fw, bc, p)

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

ctr0 = deepcopy(ctr)

cf = config_ntuple(
	t1 = 1.0,
	x0 = 0,
	x1 = 1,
	nx = 40,
	u0 = -8,
	u1 = 8,
	nu = 100,
	K = 0,
	Kn = 0.0001,
)

set = Setup(case = "wave", space = "1d0f0v", maxTime = cf.t1, limiter = "linear", flux = "hll")
ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = nothing
gas = Gas(Kn = cf.Kn, K = cf.K, γ = 3)

fw = function (x, p)
	ρ = 1.0 + 0.1 * sin(2π * x)
	u = 1.0
	t = 2 / ρ

	return prim_conserve([ρ, u, 1 / t], 3)
end
bc = function (x, p)
	w = fw(x, p)
	return conserve_prim(w, 3)
end

ib = IB(fw, bc, p)

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

sol = extract_sol(ps, ctr)
sol0 = extract_sol(ps0, ctr0)

fitp = KB.itp.interp1d(ps0.x[1:ps0.nx], sol0[:, 1], kind = "cubic")
x, dx = ps.x[1:ps.nx], ps.dx[1]
begin
    KB.L1_error(sol[:, 1], fitp(x), dx) |> println
    KB.L2_error(sol[:, 1], fitp(x), dx) |> println
    KB.L∞_error(sol[:, 1], fitp(x), dx) |> println
end

"""
0.1,
0.01896564156569092
0.00675220730284689
0.002930426171321843
0.05,
0.003971548373924417
0.0009879209567040183
0.0003132545435807655
0.025,
0.000871359641752853
0.00015317739231255725
3.470927395148316e-5

"""

KB.convergence_order(0.003971548373924417, 0.000871359641752853)
