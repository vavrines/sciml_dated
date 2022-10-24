using KitBase
#using CairoMakie, NipponColors
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)

#--- Ma=2 ---#
cf = config_ntuple(
    t1 = 250,
    x0 = -35,
    x1 = 35,
    nx = 100,
    u0 = -8,
    u1 = 8,
    nu = 80,
    v0 = -8,
    v1 = 8,
    nv = 28,
    w0 = -8,
    w1 = 8,
    nw = 28,
    nm = 5,
    K = 0,
    Kn = 1,
    Ma = 2,
    Pr = 1,
)

set = Setup(case = "shock", space = "1d1f3v", collision = "bgk", maxTime = cf.t1)
ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = VSpace3D(cf.u0, cf.u1, cf.nu, cf.v0, cf.v1, cf.nv, cf.w0, cf.w1, cf.nw)
gas = Gas(Kn = cf.Kn, Ma = cf.Ma, Pr = cf.Pr, K = cf.K)

fw, ff, bc, p = KitBase.ib_rh(set, ps, vs, gas)
ib = IB1F(fw, ff, bc, p)

ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)

dt = timestep(ks, ctr, 0)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(5)
@showprogress for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if maximum(res) < 5.e-7
        break
    end
end

fname = "sol3d_ma" * "$(cf.Ma)" * "_bgk.jld2"
@save fname ks ctr

#--- Ma=3 ---#
cf = config_ntuple(
    t1 = 250,
    x0 = -35,
    x1 = 35,
    nx = 100,
    u0 = -10,
    u1 = 10,
    nu = 80,
    v0 = -10,
    v1 = 10,
    nv = 28,
    w0 = -10,
    w1 = 10,
    nw = 28,
    nm = 5,
    K = 0,
    Kn = 1,
    Ma = 3,
    Pr = 1,
)
set = Setup(case = "shock", space = "1d1f3v", collision = "bgk", maxTime = cf.t1)
ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = VSpace3D(cf.u0, cf.u1, cf.nu, cf.v0, cf.v1, cf.nv, cf.w0, cf.w1, cf.nw)
gas = Gas(Kn = cf.Kn, Ma = cf.Ma, Pr = cf.Pr, K = cf.K)
fw, ff, bc, p = KitBase.ib_rh(set, ps, vs, gas)
ib = IB1F(fw, ff, bc, p)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)

dt = timestep(ks, ctr, 0)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(5)
@showprogress for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if maximum(res) < 5.e-7
        break
    end
end

fname = "sol3d_ma" * "$(cf.Ma)" * "_bgk.jld2"
@save fname ks ctr
