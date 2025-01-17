using KitBase
#using CairoMakie, NipponColors
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)

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
    Pr = 2 / 3,
)

set = Setup(case = "shock", space = "1d1f3v", collision = "fsm", maxTime = cf.t1)

ps = PSpace1D(cf.x0, cf.x1, cf.nx, 1)
vs = VSpace3D(cf.u0, cf.u1, cf.nu, cf.v0, cf.v1, cf.nv, cf.w0, cf.w1, cf.nw)

μᵣ = ref_vhs_vis(cf.Kn, cf.α, cf.ω)
fsm = fsm_kernel(vs, μᵣ, cf.nm, cf.α)
gas = Gas(Kn = cf.Kn, Ma = cf.Ma, Pr = cf.Pr, K = cf.K, fsm = fsm)

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

    if iter % 500 == 0
        @save "sol3d.jld2" ks ctr
    end
    if maximum(res) < 5.e-7
        break
    end
end

fname = "sol3d_ma" * "$(cf.Ma)" * ".jld2"
@save fname ks ctr
