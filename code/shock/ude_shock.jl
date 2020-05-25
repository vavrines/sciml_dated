# ------------------------------------------------------------
# Normal shock structure
# ------------------------------------------------------------

using Revise
using DifferentialEquations
using Flux
using DiffEqFlux
using Optim
using Plots
using FileIO
using JLD2
using OffsetArrays

#using Kinetic
include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
using .Kinetic

# load variables
cd(@__DIR__)
D = read_dict("ude_shock.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

# define solver settings
γ = heat_capacity_ratio(inK, 1)
set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)
μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = GasProperty(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)
vSpace = VSpace1D(umin, umax, nu, vMeshType)
vSpace3D = VSpace3D(umin, umax, nu, vmin, vmax, nv, wmin, wmax, nw, vMeshType)
wL, primL, hL, bL, bcL, wR, primR, hR, bR, bcR = ib_rh(mach, γ, vSpace.u, inK)
ib = IB1D2F(wL, primL, hL, bL, bcL, wR, primR, hR, bR, bcR)
ks = SolverSet(set, pSpace, vSpace, gas, ib, pwd())

kn_bzm = hs_boltz_kn(ks.gas.μᵣ, 1.0)
sos = sound_speed(ks.ib.primR, γ)
vmax = ks.vSpace.u1 + sos
tmax = vmax / ks.pSpace.dx[1]
dt = Float32(ks.set.cfl / tmax)
tspan = (0.f0, dt)
tran = range(tspan[1], tspan[2], length = tLen)

# initialize control volumes
ctr = OffsetArray{ControlVolume1D2F}(undef, eachindex(ks.pSpace.x))
face = Array{Interface1D2F}(undef, ks.pSpace.nx + 1)
for i in eachindex(ctr)
    if i <= ks.pSpace.nx ÷ 2
        ctr[i] = ControlVolume1D2F(
            ks.pSpace.x[i],
            ks.pSpace.dx[i],
            Float32.(ks.ib.wL),
            Float32.(ks.ib.primL),
            Float32.(ks.ib.hL),
            Float32.(ks.ib.bL),
        )
    else
        ctr[i] = ControlVolume1D2F(
            ks.pSpace.x[i],
            ks.pSpace.dx[i],
            Float32.(ks.ib.wR),
            Float32.(ks.ib.primR),
            Float32.(ks.ib.hR),
            Float32.(ks.ib.bR),
        )
    end
end
for i = 1:ks.pSpace.nx+1
    face[i] = Interface1D2F(ks.ib.wL, ks.ib.hL)
end

# deterministic BGK solutions
residual = Array{Float32}(undef, 3)
for iter = 1:2000
    Kinetic.evolve!(ks, ctr, face, dt)
    Kinetic.update!(ks, ctr, face, dt, residual)
end

plot_line(ks, ctr)

# Boltzmann dataset
f_full = Array{Float32}(undef, nu, nv, nw, nx)
for i = 1:nx
    f_full[:, :, :, i] .= full_distribution(
        ctr[i].h,
        ctr[i].b,
        vSpace.u,
        vSpace.weights,
        vSpace3D.v,
        vSpace3D.w,
        ctr[i].prim,
        ks.gas.γ,
    )
end

phi, psi, phipsi = kernel_mode(
    nm,
    vSpace3D.u1,
    vSpace3D.v1,
    vSpace3D.w1,
    vSpace3D.du[1, 1, 1],
    vSpace3D.dv[1, 1, 1],
    vSpace3D.dw[1, 1, 1],
    vSpace3D.nu,
    vSpace3D.nv,
    vSpace3D.nw,
    alphaRef,
)

function boltzmann!(df, f, p, t)
    Kn, M, phi, psi, phipsi = p
    df .= boltzmann_fft(f, Kn, M, phi, psi, phipsi)
end

data_boltz = zeros(Float32, nu, nv, nw, nx, tLen)
for i in 1:nx
    prob = ODEProblem(boltzmann!, f_full[:,:,:,i], tspan, [kn_bzm, nm, phi, psi, phipsi])
    data_boltz[:,:,:,i,:] = solve(prob, Tsit5(), saveat = tran) |> Array
end

h_boltz = zeros(Float32, nu, nx, tLen)
b_boltz = zeros(Float32, nu, nx, tLen)
for j in 1:tLen, i in 1:nx
    h_boltz[:,i,j], b_boltz[:,i,j] = reduce_distribution(
        data_boltz[:,:,:,i,j],
        vSpace3D.v,
        vSpace3D.w,
        vSpace3D.weights
    )
end
Y = vcat(h_boltz, b_boltz)

# BGK dataset
function bgk!(df, f, p, t)
    H, B, tau = p
    df[1:end÷2, :] .= (H .- f[1:end÷2, :]) ./ tau
    df[end÷2+1:end, :] .= (B .- f[end÷2+1:end, :]) ./ tau
end

X = Array{Float32}(undef, nu * 2, nx)
for i = 1:nx
    X[1:nu, i] .= ctr[i].h
    X[nu+1:end, i] .= ctr[i].b
end

H = Array{Float32}(undef, nu, nx)
B = Array{Float32}(undef, nu, nx)
τ = Array{Float32}(undef, 1, nx)
for i = 1:nx
    H[:, i] .= maxwellian(ks.vSpace.u, ctr[i].prim)
    B[:, i] .= H[:, i] .* ks.gas.K ./ (2.0 .* ctr[i].prim[end])
    τ[1, i] = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
end
P = [H, B, τ]

prob = ODEProblem(bgk!, X, tspan, P)
Y = solve(prob, Tsit5(), saveat = tran) |> Array;

#--- universal differential equation ---#
model_univ = FastChain(
    (x, p) -> x .^ 2, # initial guess
    FastDense(vSpace.nu * 2, vSpace.nu * 2 * nh, tanh),
    FastDense(vSpace.nu * 2 * nh, vSpace.nu * 2 * nh, tanh),
    FastDense(vSpace.nu * 2 * nh, vSpace.nu * 2),
)

p_model = initial_params(model_univ)
p_all = [reshape(H, :); reshape(B, :); reshape(τ, :); p_model]

function dudt_univ!(df, f, p, t)
    H = reshape(p[1:nu*nx], nu, :)
    B = reshape(p[nx*nu+1:2*nu*nx], nu, :)
    τ = reshape(p[2*nu*nx+1:2*nu*nx+nx], 1, :)
    p_nn = p[2 * nu * nx+nx+1:end]

    h = f[1:nu, :]
    b = f[nu+1:end, :]

    dh = (H .- h) ./ τ .+ model_univ(f, p_nn)[1:nu, :]
    db = (B .- b) ./ τ .+ model_univ(f, p_nn)[nu+1:end, :]

    df[1:nu, :] .= dh
    df[nu+1:end, :] .= db
end

prob_univ = ODEProblem(dudt_univ!, X, tspan, p_all)

function loss_ude(p)
    sol_univ = concrete_solve(prob_univ, Tsit5(), X, p, saveat = tran)
    loss = sum(abs2, Array(sol_univ) .- Y)
    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res = DiffEqFlux.sciml_train(loss_ude, p_all, ADAM(), cb = cb, maxiters = 200)

sol_ude = concrete_solve(prob_univ, Tsit5(), X, res.minimizer, saveat = tran)

plot(vSpace.u, sol_ude.u[1][1:nu, 25])
plot!(vSpace.u, sol_ude.u[2][1:nu, 25])
plot!(vSpace.u, sol_ude.u[3][1:nu, 25])
