using Kinetic, Solaris
using Kinetic.KitBase.Distributions, Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: Adam, Data, throttle
using Solaris.Optim: LBFGS

cd(@__DIR__)

isNewRun = true
#isNewRun = false

set = (
    u0 = -5,
    u1 = 5,
    nu = 80,
    v0 = -5,
    v1 = 5,
    nv = 28,
    w0 = -5,
    w1 = 5,
    nw = 28,
    nm = 5,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

m = moment_basis(vs3.u[:], vs3.v[:], vs3.w[:], 4)

pf = Normal(0.0, 0.01)
pn = Uniform(0.1, 10)
pt = Uniform(0.1, 8)

pdfs = []
for iter = 1:10000
    _f = sample_pdf(m, 4, [rand(pn), 0, 1/rand(pt)], pf)
    push!(pdfs, _f)
end