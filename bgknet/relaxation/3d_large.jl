using Kinetic, Solaris, OrdinaryDiffEq, Plots
using Kinetic.KitBase.Distributions, Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam
using Solaris.Optim: LBFGS
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

cd(@__DIR__)

set = (
    maxTime = 3,
    tnum = 16,
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
    Kn = 1,
    K = 0,
    alpha = 1.f0,
    omega = 0.f5,
    nh = 8,
)

vs = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw; precision = Float32)
vs1d = VSpace1D(set.u0, set.u1, set.nu; precision = Float32)
m = moment_basis(vs1d.u, 5)

pf = Normal(0.0, 0.01)
pn = Uniform(0.1, 10)
pt = Uniform(0.1, 8)

pdf1ds = []
for iter = 1:10000
    _f = sample_pdf(m, [rand(pn), 0, 1/rand(pt)], pf)
    push!(pdf1ds, _f)
end

pdfs = []
f0 = zeros(Float32, vs.nu, vs.nv, vs.nw)
@showprogress for iter in eachindex(pdf1ds)
    @threads for k = 1:vs.nw
        for j = 1:vs.nv, i = 1:vs.nu
            @inbounds f0[i, j, k] = pdf1ds[iter][i] * exp(-vs.v[i, j, k]^2) * exp(-vs.w[i, j, k]^2)
        end
    end
    push!(pdfs, f0)
end

pk = Uniform(0.001, 1)
kns = rand(pk, length(pdfs)) .|> Float32

X = Array{Float32}(undef, vs.nu, length(pdfs))
for i in axes(X, 2)
    X[:, i] .= reduce_distribution(pdfs[i], vs.weights[1, :, :])
end

τ = Array{Float32}(undef, 1, size(X, 2))
@showprogress for i in axes(τ, 2)
    μ = ref_vhs_vis(kns[i], set.alpha, set.omega)
    w = moments_conserve(pdfs[i], vs.u, vs.v, vs.w, vs.weights)
    prim = conserve_prim(w, 5/3)
    τ[1, i] = vhs_collision_time(prim, μ, set.omega)
end
X = vcat(X, τ)

phi, psi, chi = kernel_mode(
    set.nm,
    vs.u1,
    vs.v1,
    vs.w1,
    vs.du[1, 1, 1],
    vs.dv[1, 1, 1],
    vs.dw[1, 1, 1],
    vs.nu,
    vs.nv,
    vs.nw,
    set.alpha,
)

rhs3D = zeros(Float32, vs.nu, vs.nv, vs.nw, size(X, 2))
@threads for i in axes(rhs3D, 4)
    println("$i / $(length(pdfs))")
    df = @view rhs3D[:, :, :, i]
    μ = ref_vhs_vis(kns[i], set.alpha, set.omega)
    kn_bzm = hs_boltz_kn(μ, kns[i])
    boltzmann_ode!(df, pdfs[i], [kn_bzm, set.nm, phi, psi, chi], 0.0)
end

Y = Array{Float32}(undef, vs.nu, size(X, 2))
@showprogress for j in axes(rhs3D, 4)
    Y[:, j] .= reduce_distribution(rhs3D[:, :, :, j], vs.weights[1, :, :])
end

@save "dataset.jld2" X Y set

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu + 1, vs.nu * 2 + 1, tanh; bias = false), FnDense(vs.nu * 2 + 1, vs.nu, sigmoid; bias = false))

nn = BGKNet(mn, νn)
p = init_params(nn)

data = (X, Y)
L = size(data[1], 2)
loss(p) = sum(abs2, nn(data[1], p, vs1d) - data[2]) / L

his = []
cb = function (p, l)
    println("loss: $(loss(p))")
    push!(his, l)
    return false
end

res = sci_train(loss, p, Adam(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 5000)
res = sci_train(loss, res.u, LBFGS(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 1000)

cd(@__DIR__)
u = res.u
@save "minimizer.jld2" u




