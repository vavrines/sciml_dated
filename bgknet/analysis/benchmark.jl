using KitBase
using KitBase.Distributions
using BenchmarkTools

#--- wave ---#
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

prims = []
@time begin
    @showprogress for loop = 1:10
        begin
            p = (
                u = vs.u,
            )

            fw = function(x, p)
                ρ = rand(Uniform(0.5, 1.5)) * (1 + rand(Uniform(0.0, 0.2)) * sin(2π * x))
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

            if iter % 10000 == 0
                for i = 1:ks.ps.nx
                    push!(prims, ctr[i].prim)
                end
            end
        end
    end
    Xa = hcat(prims...)
end

"""
197.118049 seconds (1.49 G allocations: 485.946 GiB, 56.11% gc time, 0.38% compilation time)
"""

#--- sod ---#
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
@time @showprogress for loop = 1:10
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
    Xs = hcat(prims...)
end

"""
41.455028 seconds (41.77 M allocations: 13.149 GiB, 6.68% gc time, 87.76% compilation time)
"""

#--- sampler ---#
vs = vs = VSpace1D(-8, 8, 80)
m = moment_basis(vs.u, 4)
pf = Normal(0.0, 0.005)
pn = Uniform(0.01, 2)
pv = Uniform(-1.5, 1.5)
pt = Uniform(0.01, 4.0)

@time begin
    pdfs = []
    for iter = 1:4000
        _f = sample_pdf(m, 4, [1, rand(pv), 1/rand(pt)], pf) .* rand(pn)
        push!(pdfs, _f)
    end
end

"""
0.025768 seconds (72.73 k allocations: 13.329 MiB, 47.78% compilation time)
"""
