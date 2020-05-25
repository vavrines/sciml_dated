# ------------------------------------------------------------
# Homogeneous relaxation: universal differential equation
# ------------------------------------------------------------

cd(@__DIR__)

using Revise
using Kinetic
using DifferentialEquations, Flux
using DiffEqFlux, Optim
using Plots, Dates
using FileIO, JLD2
using Printf


#--- ground-truth data ---#

# generate parameters
config = "ude_config.txt"
D = read_dict(config)
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

tspan = (0.0, maxTime)
tran = range(tspan[1], tspan[2], length = tlen)

γ = heat_capacity_ratio(inK, 3)
vSpace = VSpace3D(u0, u1, nu, v0, v1, nv, w0, w1, nw, vMeshType)

f0 =
    Float32.(
        0.5 * (1 / π)^1.5 .*
        (exp.(-(vSpace.u .- 0.99) .^ 2) .+ exp.(-(vSpace.u .+ 0.99) .^ 2)) .*
        exp.(-vSpace.v .^ 2) .* exp.(-vSpace.w .^ 2),
    ) |> Array
prim0 = conserve_prim(
    moments_conserve(f0, vSpace.u, vSpace.v, vSpace.w, vSpace.weights),
    γ,
)
M0 = Float32.(maxwellian(vSpace.u, vSpace.v, vSpace.w, prim0)) |> Array

mu_ref = ref_vhs_vis(knudsen, alpha, omega)
kn_bzm = hs_boltz_kn(mu_ref, 1.0)
τ = Float32(vhs_collision_time(prim0, mu_ref, 0.5))

phi, psi, phipsi = kernel_mode(
    nm,
    vSpace.u1,
    vSpace.v1,
    vSpace.w1,
    vSpace.du[1, 1, 1],
    vSpace.dv[1, 1, 1],
    vSpace.dw[1, 1, 1],
    vSpace.nu,
    vSpace.nv,
    vSpace.nw,
    alpha,
);

function boltzmann!(df, f::Array{<:Real,3}, p, t)
    Kn, M, phi, psi, phipsi = p
    df .= boltzmann_fft(f, Kn, M, phi, psi, phipsi) # https://github.com/vavrines/Kinetic.jl
end

function bgk!(df, f::Array{<:Real,3}, p, t)
    g, tau = p
    df .= (g .- f) ./ tau
end

# Boltzmann
prob = ODEProblem(boltzmann!, f0, tspan, [kn_bzm, nm, phi, psi, phipsi])
data_boltz = solve(prob, Tsit5(), saveat = tran) |> Array

# BGK
prob1 = ODEProblem(bgk!, f0, tspan, [M0, τ])
data_bgk = solve(prob1, Tsit5(), saveat = tran) |> Array

data_boltz_1D = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
data_bgk_1D = zeros(Float32, axes(data_bgk, 1), axes(data_bgk, 4))
for j in axes(data_boltz_1D, 2), i in axes(data_boltz_1D, 1)
    data_boltz_1D[
        i,
        j,
    ] = sum(@. vSpace.weights[i, :, :] * data_boltz[i, :, :, j])
    data_bgk_1D[i, j] = sum(@. vSpace.weights[i, :, :] * data_bgk[i, :, :, j])
end

f0_1D = zeros(Float32, axes(f0, 1))
for i in axes(data_boltz_1D, 1)
    f0_1D[i] = sum(@. vSpace.weights[i, :, :] * f0[i, :, :])
end

M0_1D = zeros(Float32, axes(M0, 1))
for i in axes(M0, 1)
    M0_1D[i] = sum(@. vSpace.weights[i, :, :] * M0[i, :, :])
end


#--- universal differential equation ---#
model_univ = FastChain(
    (x, p) -> x .^ 2, # initial guess
    FastDense(vSpace.nu, vSpace.nu * nh, tanh),
    FastDense(vSpace.nu * nh, vSpace.nu * nh, tanh),
    FastDense(vSpace.nu * nh, vSpace.nu),
)

p_model = initial_params(model_univ)

p_all = [f0_1D; M0_1D; τ; p_model]

function dudt_univ!(df, f, p, t)
    M = p[nu+1:2*nu]
    τ = p[2*nu+1]
    p_nn = p[2*nu+2:end]
    df .= (M .- f) ./ τ .+ model_univ(f, p_nn)
end

prob_univ = ODEProblem(dudt_univ!, f0_1D, tspan, p_all)

function loss_ude(p)
    sol_univ = concrete_solve(prob_univ, Tsit5(), p[1:nu], p, saveat = tRange)

    loss = sum(abs2, Array(sol_univ) .- data_boltz_1D)

    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res = DiffEqFlux.sciml_train(
    loss_ude,
    p_all,
    ADAM(),
    cb = cb,
    maxiters = 200,
)

res = DiffEqFlux.sciml_train(
    loss_ude,
    res.minimizer,
    ADAM(),
    cb = cb,
    maxiters = 200,
)

res = DiffEqFlux.sciml_train(
    loss_ude,
    res.minimizer,
    LBFGS(),
    cb = cb,
    maxiters = 200,
)


#--- visualization ---#
sol_univ =
    concrete_solve(prob_univ, Tsit5(), f0_1D, res.minimizer, saveat = tRange)

plot(
    vSpace.u[:, vSpace.nv÷2, vSpace.nw÷2],
    data_boltz_1D[:, :],
    lw = 2,
    label = "Boltzmann",
    color = :brown,
)
plot!(
    vSpace.u[:, vSpace.nv÷2, vSpace.nw÷2],
    data_bgk_1D[:, :],
    lw = 2,
    label = "BGK",
    line = :dash,
    color = :gray32,
)
plot!(
    vSpace.u[:, vSpace.nv÷2, vSpace.nw÷2],
    Array(sol_univ),
    lw = 2,
    label = "UDE",
    line = :dash,
)


#--- test ---#

f1 = f0 ./ 2
prim1 = conserve_prim(
    moments_conserve(f1, vSpace.u, vSpace.v, vSpace.w, vSpace.weights),
    γ,
)
M1 = Float32.(maxwellian(vSpace.u, vSpace.v, vSpace.w, prim1)) |> Array
τ1 = Float32(vhs_collision_time(prim1, mu_ref, 0.5))
f1_1D = zeros(Float32, axes(f1, 1))
for i in axes(f1_1D, 1)
    f1_1D[i] = sum(@. vSpace.weights[i, :, :] * f1[i, :, :])
end
M1_1D = zeros(Float32, axes(M1, 1))
for i in axes(M1_1D, 1)
    M1_1D[i] = sum(@. vSpace.weights[i, :, :] * M1[i, :, :])
end


sol_univ1 =
    concrete_solve(prob_univ, Tsit5(), f1_1D, res.minimizer, saveat = tRange)

plot(
    vSpace.u[:, vSpace.nv÷2, vSpace.nw÷2],
    Array(sol_univ1),
    lw = 2,
    label = "UDE",
    line = :dash,
)
