using Kinetic, BenchmarkTools
using Flux: elu, relu

cd(@__DIR__)
include("../nn.jl")

@load "sol3d_ma2.jld2" ks ctr

@benchmark begin
    df = zero(ctr[1].f)
    p = (
        ks.gas.fsm.Kn,
        ks.gas.fsm.nm,
        ks.gas.fsm.ϕ,
        ks.gas.fsm.ψ,
        ks.gas.fsm.χ,
    )
    boltzmann_ode!(df, ctr[1].f, p, 0)
end

"""
BenchmarkTools.Trial: 151 samples with 1 evaluation.
    Range (min … max):  30.874 ms … 43.351 ms  ┊ GC (min … max): 0.00% … 8.48%
    Time  (median):     32.114 ms              ┊ GC (median):    0.00%
    Time  (mean ± σ):   33.102 ms ±  2.186 ms  ┊ GC (mean ± σ):  3.23% ± 4.70%

        ▁▇█ ▇▃ ▃▃▃▃▁                                              
    ▆▄▃██████▄██████▄▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▇▇▄▆▆▁▃▄▄█▆▄▆▄▃▆ ▃
    30.9 ms         Histogram: frequency by time        37.1 ms <

    Memory estimate: 102.98 MiB, allocs estimate: 1446.
"""

@load "sol1d_ma2.jld2" ks ctr

@benchmark begin
    df = zero([ctr[1].h; ctr[1].b])
    w = moments_conserve(ctr[1].h, ctr[1].b, ks.vs.u, ks.vs.weights)
    prim = conserve_prim(w, ks.gas.γ)
    M = maxwellian(ks.vs.u, prim)
    B = energy_maxwellian(M, prim, ks.gas.K)
    τ = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    p = ([M; B], τ)
    bgk_ode!(df, [ctr[1].h; ctr[1].b], p, 0)
end

"""
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
    Range (min … max):  3.169 μs …  3.788 ms  ┊ GC (min … max):  0.00% … 99.68%
    Time  (median):     5.893 μs              ┊ GC (median):     0.00%
    Time  (mean ± σ):   6.774 μs ± 63.692 μs  ┊ GC (mean ± σ):  16.26% ±  1.73%

                                        ▃▅███▆▄▅▃▃               
    ▃▅▇█▇▅▄▃▂▃▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▃▃▅▇███████████▆▅▅▅▅▅▄▄▃▃▃▂ ▄
    3.17 μs        Histogram: frequency by time        7.17 μs <

    Memory estimate: 12.92 KiB, allocs estimate: 36.
"""

nm = ks.vs.nu * 2
nν = nm + 1
mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm; bias = false))
νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
nn = BGKNet(mn, νn)
u = init_params(nn)
f = Float32[ctr[1].h; ctr[1].b; 1.0]

@benchmark nn(f, u, ks.vs, 5/3)
#@benchmark nn(f, u, ks.vs, 5/3, VDF{2,1}, Class{2})

"""
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
    Range (min … max):  252.897 μs …   5.640 ms  ┊ GC (min … max):  0.00% … 91.32%
    Time  (median):     295.947 μs               ┊ GC (median):     0.00%
    Time  (mean ± σ):   332.616 μs ± 394.974 μs  ┊ GC (mean ± σ):  10.04% ±  7.89%

            ▃█▆▆█▆▂                                                 
    ▂▂▃▃▃▃▄███████▆▄▃▃▃▃▂▂▂▂▂▂▂▁▁▁▂▂▂▂▂▂▁▂▁▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂ ▃
    253 μs           Histogram: frequency by time          495 μs <

    Memory estimate: 2.38 MiB, allocs estimate: 93.
"""
