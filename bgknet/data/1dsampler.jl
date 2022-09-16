using KitBase
using KitBase.Distributions, KitBase.JLD2
using Base.Threads: @threads

cd(@__DIR__)

set = config_ntuple(
    u0 = -5,
    u1 = 5,
    nu = 80,
    K = 0,
    datasize = 10000,
    isUpwind = false,
)

vs = VSpace1D(set.u0, set.u1, set.nu)

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

X = Array{Float64}(undef, vs.nu, length(pdfs))
for i in axes(X, 2)
    @assert moments_conserve(X[:, i], vs.u, vs.weights)[1] < 50
    X[:, i] .= pdfs[i]
end

pk = Uniform(0.001, 1)
kns = rand(pk, length(pdfs))

τ = Array{Float64}(undef, 1, size(X, 2))
for i in axes(τ, 2)
    μ = ref_vhs_vis(kns[i], set.α, set.ω)
    w = moments_conserve(pdfs[i], vs.u, vs.weights)
    prim = conserve_prim(w, 3)
    τ[1, i] = vhs_collision_time(prim, μ, set.ω)
end
X = vcat(X, τ)

Y = Array{Float64}(undef, vs.nu, size(X, 2))
for j in axes(Y, 2)
    w = moments_conserve(pdfs[j], vs.u, vs.weights)
    prim = conserve_prim(w, 3)
    M = maxwellian(vs.u, prim)
    q = heat_flux(pdfs[j], prim, vs.u, vs.weights)
    S = shakhov(vs.u, M, q, prim, 2/3) # Pr shouldn't be 0
    @. Y[:, j] = (M + S - pdfs[j]) / τ[1, j]
end

@save "shakhov1d.jld2" X Y
