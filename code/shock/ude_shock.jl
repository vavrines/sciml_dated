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
vSpace2D = VSpace2D(vmin, vmax, nv, wmin, wmax, nw, vMeshType)
vSpace3D = VSpace3D(umin, umax, nu, vmin, vmax, nv, wmin, wmax, nw, vMeshType)
wL, primL, hL, bL, bcL, wR, primR, hR, bR, bcR = ib_rh(mach, γ, vSpace.u, inK)
ib = IB1D2F(wL, primL, hL, bL, bcL, wR, primR, hR, bR, bcR)
ks = SolverSet(set, pSpace, vSpace, gas, ib, pwd())

kn_bzm = hs_boltz_kn(ks.gas.μᵣ, 1.0)
sos = sound_speed(ks.ib.primR, γ)
tmax = (ks.vSpace.u1 + sos) / ks.pSpace.dx[1]
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


#--- Boltzmann dataset ---#
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
for i = 1:nx
    prob = ODEProblem(
        boltzmann!,
        f_full[:, :, :, i],
        tspan,
        [kn_bzm, nm, phi, psi, phipsi],
    )
    data_boltz[:, :, :, i, :] = solve(prob, Tsit5(), saveat = tran) |> Array
end

h_boltz = zeros(Float32, nu, nx, tLen)
b_boltz = zeros(Float32, nu, nx, tLen)
for j = 1:tLen, i = 1:nx
    h_boltz[:, i, j], b_boltz[:, i, j] = reduce_distribution(
        data_boltz[:, :, :, i, j],
        vSpace3D.v,
        vSpace3D.w,
        vSpace2D.weights,
    )
end
Y = vcat(h_boltz, b_boltz)


#--- BGK dataset ---#
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
Y1 = solve(prob, Midpoint(), saveat = tran) |> Array;

plot(vSpace.u, Y[1:nu, 25, 3])
plot!(vSpace.u, Y1[1:nu, 25, 3])


#--- universal differential equation ---#
model_univ = FastChain(
    #(x, p) -> x .^ 2, # initial guess
    (x, p) -> zeros(Float32, axes(x)),
    FastDense(vSpace.nu * 2, vSpace.nu * 2 * nh, tanh),
    #FastDense(vSpace.nu * 2 * nh, vSpace.nu * 2 * nh, tanh),
    FastDense(vSpace.nu * 2 * nh, vSpace.nu * 2),
)

p_model = initial_params(model_univ)

function dfdt!(df, f, p, t)
    h = f[1:nu, :]
    b = f[nu+1:end, :]

    dh = (H .- h) ./ τ .+ model_univ(f, p)[1:nu, :]
    db = (B .- b) ./ τ .+ model_univ(f, p)[nu+1:end, :]

    df[1:nu, :] .= dh
    df[nu+1:end, :] .= db
end

prob_ube = ODEProblem(dfdt!, X, tspan, p_model)

function loss(p)
    sol_ube = concrete_solve(prob_ube, Midpoint(), X, p, saveat = tran)
    loss = sum(abs2, Array(sol_ube) .- Y1)
    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res = DiffEqFlux.sciml_train(
    loss,
    p_model,
    ADAM(),
    cb = Flux.throttle(cb, 1),
    maxiters = 200,
)

res = DiffEqFlux.sciml_train(
    loss_ude,
    res.minimizer,
    ADAM(),
    cb = Flux.throttle(cb, 1),
    maxiters = 200,
)

sol_ude = concrete_solve(prob_univ, Tsit5(), X, res.minimizer, saveat = tran)

plot(vSpace.u, sol_ude.u[1][1:nu, 25])
plot!(vSpace.u, sol_ude.u[2][1:nu, 25])
plot!(vSpace.u, sol_ude.u[3][1:nu, 25])

model_univ(X, res.minimizer)



function nbe_rhs!(df, f, p, t)
    H = p[1:nu]
    B = p[nu+1:2*nu]
    τ = p[2*nu+1]
    p_nn = p[2*nu+2:end]

    h = f[1:nu]
    b = f[nu+1:end]

    dh = (H .- h) ./ τ .+ model_univ(f, p_nn)[1:nu]
    db = (B .- b) ./ τ .+ model_univ(f, p_nn)[nu+1:end]

    df[1:nu] .= dh
    df[nu+1:end] .= db
end

ube = ODEProblem(
    nbe_rhs!,
    [ctr[1].h; ctr[1].b],
    tspan,
    [H[:, 1]; B[:, 1]; τ[1, 1]; res.minimizer],
)

function step_nbe!(
    fwL,
    fhL,
    fbL,
    w,
    prim,
    h,
    b,
    fwR,
    fhR,
    fbR,
    K,
    γ,
    μ,
    ω,
    u,
    weights,
    p,
    dx,
    tran,
    RES,
    AVG,
)

    #--- record W^{n} ---#
    w_old = deepcopy(w)
    H = maxwellian(u, prim)
    B = H .* K ./ (2.0 .* prim[end])
    τ = vhs_collision_time(prim, μ, ω)

    #--- update W^{n+1} ---#
    @. w += (fwL - fwR) / dx
    prim .= conserve_prim(w, γ)

    #--- record residuals ---#
    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    #--- update f^{n+1} ---#
    sol = concrete_solve(ube, Midpoint(), [h; b], [H; B; τ; p], saveat = tran)
    hstar = sol.u[end][1:length(h)]
    bstar = sol.u[end][length(h)+1:end]
    for i in eachindex(h)
        h[i] = hstar[i] + (fhL[i] - fhR[i]) / dx
        b[i] = bstar[i] + (fbL[i] - fbR[i]) / dx
    end

end

sumRes = zeros(Float32, axes(ib.wL))
sumAvg = zeros(Float32, axes(ib.wL))
for iter = 1:50
    Kinetic.evolve!(ks, ctr, face, dt)

    for i = 2:49
        step_nbe!(
            face[i].fw,
            face[i].fh,
            face[i].fb,
            ctr[i].w,
            ctr[i].prim,
            ctr[i].h,
            ctr[i].b,
            face[i+1].fw,
            face[i+1].fh,
            face[i+1].fb,
            ks.gas.K,
            ks.gas.γ,
            ks.gas.μᵣ,
            ks.gas.ω,
            ks.vSpace.u,
            ks.vSpace.weights,
            res.minimizer,
            ctr[i].dx,
            tran,
            sumRes,
            sumAvg,
        )
    end
end

plot_line(ks, ctr)

plot(vSpace.u, ctr[44].h)
