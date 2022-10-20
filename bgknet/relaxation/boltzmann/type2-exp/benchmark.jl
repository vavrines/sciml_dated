using Kinetic, OrdinaryDiffEq, BenchmarkTools
using KitBase.JLD2
using Solaris.Flux: throttle, Adam, Data, elu, relu

cd(@__DIR__)
include("../../../nn.jl")

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
    t1 = 8,
    nt = 81,
    Kn = 1,
)

tspan = (0, set.t1)
tsteps = linspace(tspan[1], tspan[2], set.nt)
γ = 5 / 3

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

f0 = 0.5 * (1 / π)^1.5 .*
    (exp.(-(vs3.u .- 1) .^ 2) .+ 0.7 .* exp.(-(vs3.u .+ 1) .^ 2)) .*
    exp.(-vs3.v .^ 2) .* exp.(-vs3.w .^ 2)

mu_ref = ref_vhs_vis(set.Kn, set.α, set.ω)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)

w0 = moments_conserve(f0, vs3.u, vs3.v, vs3.w, vs3.weights)
prim0 = conserve_prim(w0, γ)
M0 = (maxwellian(vs3.u, vs3.v, vs3.w, prim0))
τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

phi, psi, chi = kernel_mode(
    set.nm,
    vs3.u1,
    vs3.v1,
    vs3.w1,
    vs3.du[1, 1, 1],
    vs3.dv[1, 1, 1],
    vs3.dw[1, 1, 1],
    vs3.nu,
    vs3.nv,
    vs3.nw,
    set.α,
)
prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, chi])
@benchmark data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

"""
BenchmarkTools.Trial: 3 samples with 1 evaluation.
    Range (min … max):  2.158 s …   2.197 s  ┊ GC (min … max): 2.61% … 2.83%
    Time  (median):     2.167 s              ┊ GC (median):    2.72%
    Time  (mean ± σ):   2.174 s ± 20.271 ms  ┊ GC (mean ± σ):  2.72% ± 0.11%

    █            █                                          █  
    █▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
    2.16 s         Histogram: frequency by time         2.2 s <

    Memory estimate: 6.43 GiB, allocs estimate: 90322.
"""

prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
@benchmark data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

data_boltz_1D = zeros(axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2)
    data_boltz_1D[:, j] .=
        reduce_distribution(data_boltz[:, :, :, j], vs2.weights)
    data_bgk_1D[:, j] .=
        reduce_distribution(data_bgk[:, :, :, j], vs2.weights)
end
h0_1D, b0_1D = reduce_distribution(f0, vs3.v, vs3.w, vs2.weights)
H0_1D, B0_1D = reduce_distribution(M0, vs3.v, vs3.w, vs2.weights)

@load "prototype2.jld2" nn u
@load "specialize.jld2" u

function dfdt(df, f, p, t)
    nn, u, vs, γ = p
    df .= nn([f; τ0], u, vs, γ, VDF{2,1}, Class{2})
end

ube = ODEProblem(dfdt, [h0_1D; b0_1D], tspan, (nn, u, vs, 5/3))
@benchmark solve(ube, Midpoint(); saveat = tsteps)

"""
BenchmarkTools.Trial: 116 samples with 1 evaluation.
    Range (min … max):  42.021 ms … 46.736 ms  ┊ GC (min … max): 8.23% … 14.80%
    Time  (median):     42.472 ms              ┊ GC (median):    8.28%
    Time  (mean ± σ):   43.189 ms ±  1.468 ms  ┊ GC (mean ± σ):  9.91% ±  2.83%

        █▃▂                                                      
    ▃▃▅████▇▆▅▃▃▃▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▄▅▄▅▅▃▃ ▃
    42 ms           Histogram: frequency by time        46.2 ms <

    Memory estimate: 375.66 MiB, allocs estimate: 14344.
"""

function bgk_ode2(f, p, t)
    df = zero(f)
    u, ω, γ, K, mu, idx = p

    h = @view f[1:end÷2]
    b = @view f[end÷2+1:end]

    w = moments_conserve(h, b, u, ω)
    prim = conserve_prim(w, γ)
    H = maxwellian(u, prim)
    B = energy_maxwellian(h, prim, K)
    M = [H; B]
    tau = vhs_collision_time(prim, mu, idx)

    df = @. (M - f) / tau
    return df
end

prob2 = ODEProblem(bgk_ode2, [h0_1D; b0_1D], tspan, (vs.u, vs.weights, 5/3, 2, mu_ref, set.ω))
@benchmark solve(prob2, Tsit5(); saveat = tsteps) |> Array

"""
BenchmarkTools.Trial: 3700 samples with 1 evaluation.
    Range (min … max):  399.565 μs … 35.279 ms  ┊ GC (min … max):  0.00% … 93.29%
    Time  (median):       1.076 ms              ┊ GC (median):     0.00%
    Time  (mean ± σ):     1.350 ms ±  3.102 ms  ┊ GC (mean ± σ):  23.22% ±  9.72%

    █                                                            
    ▄█▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▂
    400 μs          Histogram: frequency by time         29.1 ms <

    Memory estimate: 3.38 MiB, allocs estimate: 2939.
"""
