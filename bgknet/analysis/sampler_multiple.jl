"""
sampling strategy vs. multiple simulation-based generation

"""

using CairoMakie, KitBase, NipponColors
using KitBase.Distributions, KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

dc = dict_color()
tc = plot_color()
cd(@__DIR__)

#--- advection-based generation ---#
set = Setup(
    space = "1d1f1v",
    boundary = "period",
    maxTime = 2,
)
ps = PSpace1D(0, 1, 200, 1)
vs = VSpace1D(-8, 8, 80)
gas = Gas(
    Kn = 1e-3,
    K = 0,
    γ = 3,
)

prims = []
@showprogress for loop = 1:10
    begin
        p = (
            u = vs.u,
        )

        fw = function(x, p)
            ρ = rand(Uniform(0.5, 2)) * (1 + 0.1 * sin(2π * x))
            u = rand()
            λ = ρ / 2 / (rand(Uniform(0.5, 1.5)))
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

    for iter = 1:nt
        reconstruct!(ks, ctr)
        evolve!(ks, ctr, face, dt)
        update!(ks, ctr, face, dt, res)

        if iter % 3000 == 0
            for i = 1:ks.ps.nx
                push!(prims, ctr[i].prim)
            end
        end
    end
end
Xa = hcat(prims...)

#--- Sod-based generation ---#
set = Setup(
    space = "1d1f1v",
    boundary = "fix",
    maxTime = 0.15,
)
ps = PSpace1D(0, 1, 200, 1)
vs = VSpace1D(-8, 8, 80)
gas = Gas(
    Kn = 1e-4,
    K = 0,
    γ = 3,
)

prims = []
@showprogress for loop = 1:10
    begin
        primL = [rand(Uniform(0.0, 2.0)), 0.0, 1/rand(Uniform(1.0, 3.0))]
        primR = [rand(Uniform(0.0, 0.25)), 0.0, 1/rand(Uniform(0.0, 1.25))]

        wL = prim_conserve(primL, gas.γ)
        wR = prim_conserve(primR, gas.γ)

        p = (
            x0 = ps.x0,
            x1 = ps.x1,
            wL = wL,
            wR = wR,
            primL = primL,
            primR = primR,
            γ = gas.γ,
            u = vs.u,
        )

        fw = function (x, p)
            if x <= (p.x0 + p.x1) / 2
                return p.wL
            else
                return p.wR
            end
        end
        ff = function (x, p)
            w = ifelse(x <= (p.x0 + p.x1) / 2, p.wL, p.wR)
            prim = conserve_prim(w, p.γ)
            h = maxwellian(p.u, prim)
            return h
        end
        bc = function (x, p)
            if x <= (p.x0 + p.x1) / 2
                return p.primL
            else
                return p.primR
            end
        end
    end
    ib = IB1F(fw, ff, bc, p)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks)
    dt = timestep(ks, ctr, 0)
    nt = Int(floor(ks.set.maxTime / dt))
    res = zeros(3)

    for iter = 1:nt
        reconstruct!(ks, ctr)
        evolve!(ks, ctr, face, dt)
        update!(ks, ctr, face, dt, res)

        if iter % 200 == 0
            for i = 1:ks.ps.nx
                push!(prims, ctr[i].prim)
            end
        end
    end
end
Xs = hcat(prims...)

#--- sampling-based generation ---$
vs = vs = VSpace1D(-8, 8, 80)
m = moment_basis(vs.u, 4)

pf = Normal(0.0, 0.005)
pn = Uniform(0.01, 2)
pv = Uniform(-1.5, 1.5)
pt = Uniform(0.01, 4.0)

pdfs = []
for iter = 1:4000
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
    fig
end
#save("sample_t_multi.pdf", fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "ρ", ylabel = "U", title = "")
    scatter!(Xg[1, :], Xg[2, :]; color = (dc["tokiwa"], 0.7), label = "Sample")
    scatter!(Xs[1, :], Xs[2, :]; color = (dc["ukon"], 0.7), label = "Sod")
    scatter!(Xa[1, :], Xa[2, :]; color = (tc[2], 0.7), label = "Advection")
    axislegend()
    fig
end
#save("sample_u_multi.pdf", fig)

@save "multiple.jld2" Xs Xa Xg
