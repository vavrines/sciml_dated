using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
using .Kinetic

function bgk!(df, f, p, t)
    g, tau = p
    df .= (g .- f) ./ tau
end

# may need to change working directory
D = read_dict("../shock/shock1D.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

γ = 3.
set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)
μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = Gas(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)
vSpace = VSpace1D(umin, umax, nu, vMeshType, nug)
wL, primL, fL, bcL, wR, primR, fR, bcR = ib_rh(mach, γ, vSpace.u)
ib = IB1D1F(wL, primL, fL, bcL, wR, primR, fR, bcR)

sos = sound_speed(ib.primR, γ)
vmax = vSpace.u1 + sos
tmax = vmax / pSpace.dx[1]
dt = set.cfl / tmax

tSpan = (0.f0, Float32(dt)*2)
tRan = range(tSpan[1], tSpan[2], length=tLen)

ML = Float32.(maxwellian(vSpace.u, ib.primL)) |> Array
MR = Float32.(maxwellian(vSpace.u, ib.primR)) |> Array

f1 = deepcopy(ML)
f2 = @. 0.5 * ML + 0.5 * MR
f3 = deepcopy(MR)

prim1 = conserve_prim(moments_conserve( f1, vSpace.u, vSpace.weights ), γ)
prim2 = conserve_prim(moments_conserve( f2, vSpace.u, vSpace.weights ), γ)
prim3 = conserve_prim(moments_conserve( f3, vSpace.u, vSpace.weights ), γ)

M1 = maxwellian(vSpace.u, prim1)
M2 = maxwellian(vSpace.u, prim2)
M3 = maxwellian(vSpace.u, prim3)

τ1 = vhs_collision_time(prim1, μᵣ, gas.ω)
τ2 = vhs_collision_time(prim2, μᵣ, gas.ω)
τ3 = vhs_collision_time(prim3, μᵣ, gas.ω)

prob = ODEProblem(bgk!, f1, tSpan, [M1, τ1])
data_boltz1 = solve(prob, Tsit5(), saveat=tRan) |> Array;

prob = ODEProblem(bgk!, f2, tSpan, [M2, τ2])
data_boltz2 = solve(prob, Tsit5(), saveat=tRan) |> Array;

prob = ODEProblem(bgk!, f3, tSpan, [M3, τ3])
data_boltz3 = solve(prob, Tsit5(), saveat=tRan) |> Array;

d = 3
X = Array{Float32}(undef, nu, d)
X[:, 1] .= f1
X[:, 2] .= f2
X[:, 3] .= f3

Y = Array{Float32}(undef, nu, tLen, d)
Y[:, :, 1] .= data_boltz1
Y[:, :, 2] .= data_boltz2
Y[:, :, 3] .= data_boltz3

data = [(X[:, i], Y[:, :, i]) for i = 1:d]
dataset = Iterators.cycle(data)

dudt = FastChain( (x, p) -> x.^3,
                   FastDense(vSpace.nu, vSpace.nu*10, tanh),
                   FastDense(vSpace.nu*10, vSpace.nu) )
node = NeuralODE(dudt, tSpan, Tsit5(), saveat=tRan)

function loss_node(p, x, y)
    pred = node(x, p)
    loss = sum(abs2, y .- pred)
    loss
end

cb = function (p, l) #callback function to observe training
    print("loss: "); display(l)
    return false
end

res = DiffEqFlux.sciml_train(
    loss_node,
    node.p,
    ADAM(0.05),
    dataset,
    cb = cb,
    maxiters = 200,
)

res = DiffEqFlux.sciml_train(
    loss_node,
    res.minimizer,
    ADAM(),
    dataset,
    cb = cb,
    maxiters = 500,
)

res = DiffEqFlux.sciml_train(
    loss_node,
    res.minimizer,
    LBFGS(),
    dataset,
    cb = cb,
    maxiters = 200,
)

plot(vSpace.u, data_boltz2)
plot(vSpace.u, node(f2, res.minimizer).u)
