using Kinetic, Solaris, OrdinaryDiffEq, Plots
using Kinetic.KitBase.Distributions, Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu
using Solaris.Optim: LBFGS
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads

struct BNet{T1,T2,T3}
    Mnet::T1
    νnet::T2
    fn::T3
end

BNet(m, ν) = BNet(m, ν, -)

(nn::BNet)(x, p, vs = VSpace1D(-6, 6, size(x)[1] - 1; precision = Float32), γ = 3) = begin
    nm = param_length(nn.Mnet)
    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = KB.f_maxwellian(f, vs, γ)
    y = f .- M
    z = vcat(y, τ)
    
    (nn.fn(relu(M .+ nn.Mnet(y, p[1:nm])), f)) ./ (τ .+ τ ./ 5 .* tanh.(nn.νnet(z, p[nm+1:end])))
    #(nn.fn(relu(M .+ nn.Mnet(y, p[1:nm])), f)) ./ τ
end

Solaris.init_params(nn::BNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))

cd(@__DIR__)

set = (
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 0,
    alpha = 1.f0,
    omega = 0.f5,
)

vs = VSpace1D(set.u0, set.u1, set.nu; precision = Float32)
m = moment_basis(vs.u, 5)

pf = Normal(0.0, 0.01)
pn = Uniform(0.1, 10)
pt = Uniform(0.1, 8)

pdfs = []
for iter = 1:10000
    _f = sample_pdf(m, [rand(pn), 0, 1/rand(pt)], pf) .|> Float32
    push!(pdfs, _f)
end

pk = Uniform(0.001, 1)
kns = rand(pk, length(pdfs)) .|> Float32

X = Array{Float32}(undef, vs.nu, length(pdfs))
for i in axes(X, 2)
    @assert moments_conserve(pdfs[i], vs.u, vs.weights)[1] < 50
    X[:, i] .= pdfs[i]
end

τ = Array{Float32}(undef, 1, size(X, 2))
for i in axes(τ, 2)
    μ = ref_vhs_vis(kns[i], set.alpha, set.omega)
    w = moments_conserve(pdfs[i], vs.u, vs.weights)
    prim = conserve_prim(w, 3)
    τ[1, i] = vhs_collision_time(prim, μ, set.omega)
end
X = vcat(X, τ)

Y = Array{Float32}(undef, vs.nu, size(X, 2))
for j in axes(Y, 2)
    w = moments_conserve(pdfs[j], vs.u, vs.weights)
    prim = conserve_prim(w, 3)
    M = maxwellian(vs.u, prim)
    q = heat_flux(pdfs[j], prim, vs.u, vs.weights)
    S = shakhov(vs.u, M, q, prim, 1)
    #@. Y[:, j] = (S - pdfs[j]) / τ[1, j]
    @. Y[:, j] = (M - pdfs[j]) / τ[1, j]
end

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu + 1, vs.nu * 2 + 2, tanh; bias = false), FnDense(vs.nu * 2 + 2, vs.nu; bias = false))
#nn = BGKNet(mn, νn)
nn = BNet(mn, νn)
p = init_params(nn)

data = (X, Y)
L = size(data[1], 2)
loss(p) = sum(abs2, nn(data[1], p, vs) - data[2]) / L

loss(p)
loss(zero(p))

#his = []
cb = function (p, l)
    println("loss: $(loss(p))")
    #push!(his, l)
    return false
end

res = sci_train(loss, p, Adam(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 1000)
res = sci_train(loss, res.u, LBFGS(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 200)



plot(vs.u, X[1:end-1, 101])
plot(Y[:, 8])
