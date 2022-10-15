"""
sampling strategy vs. single simulation-based generation

"""

using CairoMakie, KitBase, NipponColors
using KitBase.Distributions, KitBase.JLD2

dc = dict_color()
tc = plot_color()
cd(@__DIR__)

#--- advection-based generation ---#
set = Setup(
    space = "1d1f1v",
    boundary = "period",
    maxTime = 5,
)
ps = PSpace1D(0, 1, 200, 1)
vs = VSpace1D(-8, 8, 80)
gas = Gas(
    Kn = 1e-3,
    K = 0,
    γ = 3,
)
begin
    p = (
        u = vs.u,
    )

    fw = function(x, p)
        ρ = 1 + 0.1 * sin(2π * x)
        u = 1.0
        λ = ρ / 2
        prim_conserve([ρ, u, λ], 3)
    end
    bc = function (x, p)
        w = fw(x, p)
        conserve_prim(w, 3)
    end
    ff = function (x, p)
        w = fw(x, p)
        prim = conserve_prim(w, 3)
        maxwellian(p.u, prim)
    end
end
ib = IB1F(fw, ff, bc, p)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)
dt = timestep(ks, ctr, 0)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(3)

prims = []
for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter % 1200 == 0
        for i = 1:ks.ps.nx
            push!(prims, ctr[i].prim)
        end
    end
end
Xa = hcat(prims...)

#--- Sod-based generation ---#
ks, ctr, face, t = initialize("sod.txt")
dt = timestep(ks, ctr, t)
nt = Int(floor(ks.set.maxTime / dt))
res = zeros(3)

prims = []
for iter = 1:nt
    reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter % 32 == 0
        for i = 1:ks.ps.nx
            push!(prims, ctr[i].prim)
        end
    end
end
Xs = hcat(prims...)

#--- sampling-based generation ---$
vs = vs = VSpace1D(-8, 8, 80)
m = moment_basis(vs.u, 4)

pf = Normal(0.0, 0.005)
pn = Uniform(0.01, 2)
pv = Uniform(-1, 1)
pt = Uniform(0.01, 3.0)

pdfs = []
for iter = 1:3000
    _f = sample_pdf(m, 4, [1, rand(pv), 1/rand(pt)], pf)
    _f .= _f .* rand(pn)
    if moments_conserve(_f, vs.u, vs.weights)[1] < 10
        push!(pdfs, _f)
    end
end

pk = Uniform(0.001, 1)
kns = rand(pk, length(pdfs))

X = Array{Float64}(undef, vs.nu, length(pdfs))
for i in axes(X, 2)
    @assert moments_conserve(pdfs[i], vs.u, vs.weights)[1] < 50
    X[:, i] .= pdfs[i]
end

Xg = Array{Float64}(undef, 3, size(X, 2))
for j in axes(Xg, 2)
    w = moments_conserve(X[:, j], vs.u, vs.weights, VDF{1,1})
    Xg[:, j] .= conserve_prim(w, 3)
end

#--- plot ---#
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "ρ", ylabel = "T", title = "")
    scatter!(Xg[1, :], 1 ./ Xg[end, :]; color = (dc["tokiwa"], 0.7), label = "Sample")
    scatter!(Xs[1, :], 1 ./ Xs[end, :]; color = (dc["ukon"], 0.7), label = "Sod")
    scatter!(Xa[1, :], 1 ./ Xa[end, :]; color = (tc[2], 0.7), label = "Advection")
    axislegend()
#    xlims!(ax, [-0.1, 2.1])
#    ylims!(ax, (-0.1, 3.1))
    fig
end
save("sample_t.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "ρ", ylabel = "U", title = "")
    scatter!(Xg[1, :], Xg[2, :]; color = (dc["tokiwa"], 0.7), label = "Sample")
    scatter!(Xs[1, :], Xs[2, :]; color = (dc["ukon"], 0.7), label = "Sod")
    scatter!(Xa[1, :], Xa[2, :]; color = (tc[2], 0.7), label = "Advection")
    axislegend()
    fig
end
save("sample_u.pdf", fig)
