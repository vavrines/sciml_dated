using KitBase
using KitBase.Distributions, KitBase.JLD2
using Base.Threads: @threads

cd(@__DIR__)

set = config_ntuple(
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
    Kn = 1,
    datasize = 10000,
    isUpwind = false,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

m = moment_basis(vs.u, 4)

pf = Normal(0.0, 0.005)
pn = Uniform(0.1, 2)
pt = Uniform(0.5, 2)

pdfs = []
for iter = 1:set.datasize
    _f = sample_pdf(m, 4, [rand(pn), 0, 1/rand(pt)], pf)
    push!(pdfs, _f)
end

δ = heaviside.(vs.u)
pdf1s = []
for i = 1:length(pdfs)÷2
    _f = pdfs[i] .* δ + pdfs[length(pdfs) + 1 - i] .* (1 .- δ)
    push!(pdf1s, _f)
end

if set.isUpwind
    pdfs = pdf1s
end

X0 = Array{Float64}(undef, vs3.nu, vs3.nv, vs3.nw, length(pdfs))
@threads for iter in axes(X0)[end]
    w = moments_conserve(pdfs[iter], vs.u, vs.weights)
    prim = conserve_prim(w, 3)
    for i = 1:vs.nu
        X0[i, :, :, iter] .= pdfs[iter][i] * exp.(-vs3.v[i, :, :] .^ 2) .* exp.(-vs3.w[i, :, :] .^ 2)
    end
end

X = Array{Float64}(undef, vs.nu*2, length(pdfs))
for i in axes(X, 2)
    @assert moments_conserve(X0[:, :, :, i], vs3.u, vs3.v, vs3.w, vs3.weights)[1] < 50
    X[1:vs.nu, i], X[vs.nu+1:end, i] = reduce_distribution(X0[:, :, :, i], vs3.v, vs3.w, vs2.weights)
end

pk = Uniform(0.001, 1)
kns = rand(pk, length(pdfs))

ϕ, ψ, χ = kernel_mode(5, vs3.u1, vs3.v1, vs3.w1, vs3.du[1], vs3.dv[1], vs3.dw[1],
    vs3.nu, vs3.nv, vs3.nw, set.α)

Y0 = Array{Float64}(undef, vs3.nu, vs3.nv, vs3.nw, length(pdfs))
@threads for j in axes(Y0)[end]
    μ = ref_vhs_vis(kns[j], set.α, set.ω)
    kn_bz = hs_boltz_kn(μ, set.α)

    Q = @view Y0[:, :, :, j]
    boltzmann_fft!(Q, X0[:, :, :, j], kn_bz, 5, ϕ, ψ, χ)
end

Y = Array{Float64}(undef, vs.nu*2, length(pdfs))
for i in axes(Y, 2)
    Y[1:vs.nu, i], Y[vs.nu+1:end, i] = reduce_distribution(Y0[:, :, :, i], vs3.v, vs3.w, vs2.weights)
end

τ = Array{Float64}(undef, 1, size(X, 2))
for i in axes(τ, 2)
    μ = ref_vhs_vis(kns[i], set.α, set.ω)
    w = moments_conserve(X[1:vs.nu, i], X[vs.nu+1:end, i], vs.u, vs.weights)
    prim = conserve_prim(w, 5/3)
    τ[1, i] = vhs_collision_time(prim, μ, set.ω)
end
X = vcat(X, τ)

@save "boltz3d.jld2" X Y
