using KitBase
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)

cf = config_ntuple(
    t1 = 250,
    x0 = -35,
    x1 = 35,
    nx = 100,
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 2,
    Kn = 1,
    #Ma = 2,
    Ma = 3,
    Pr = 1,
)

set = Setup(case = "shock", space = "1d2f1v", collision = "bgk", maxTime = cf.t1)
ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = VSpace1D(cf.u0, cf.u1, cf.nu)
gas = Gas(Kn = cf.Kn, Ma = cf.Ma, Pr = cf.Pr, K = cf.K)

fw, ff, bc, p = KitBase.ib_rh(set, ps, vs, gas)
ib = IB2F(fw, ff, bc, p)

ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)

dt = timestep(ks, ctr, 0)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(3)

@showprogress for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if maximum(res) < 5.e-7
        break
    end
end

fname = "sol1d_ma" * "$(cf.Ma)" * ".jld2"
@save fname ks ctr
