using KitBase, Solaris, OrdinaryDiffEq
using KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Flux: elu, relu, Data, Adam, throttle, sigmoid

function invariant_pdf(α::AbstractVector, vs)
    return exp.(α[1] .+ α[2] .* vs.u .+ α[3] .* vs.u .^ 2 ./ 2)
end

function invariant_pdf(α::AbstractMatrix, vs)
    M = [invariant_pdf(α[:, j], vs) for j in axes(α, 2)]

    return hcat(M...)
end

set = (
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
    t1 = 8,
    tnum = 50,
    Kn = 1,
)

vs = VSpace1D(set.u0, set.u1, set.nu)

begin
    tspan = (0, set.t1)
    tsteps = linspace(tspan[1], tspan[2], set.tnum)
    γ = 3.0
    vs = VSpace1D(set.u0, set.u1, set.nu)
    momentquad = vs.u .* vs.weights
    energyquad = vs.u .^ 2 .* vs.weights

    f0 = @. 0.5 * (1 / π)^0.5 * (exp.(-(vs.u - 2) ^ 2) + 0.5 * exp(-(vs.u + 2) ^ 2))
    prim0 = conserve_prim(moments_conserve(f0, vs.u, vs.weights), γ)
    M0 = maxwellian(vs.u, prim0)

    mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    q = heat_flux(f0, prim0, vs.u, vs.weights)
    S0 = shakhov(vs.u, M0, q, prim0, 2 / 3)

    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    prob2 = ODEProblem(bgk_ode!, f0, tspan, [M0 .+ S0, τ0])
    data_shakhov = solve(prob2, Tsit5(), saveat = tsteps) |> Array
end

X1 = Array{Float64}(undef, vs.nu, size(data_shakhov, 2))
for i in axes(X1, 2)
    X1[:, i] .= data_shakhov[:, i]
end
τ = Array{Float64}(undef, 1, size(X1, 2))
for i in axes(τ, 2)
    τ[1, i] = τ0
end
X1 = vcat(X1, ones(Float64, 1, size(X1, 2)) .* τ[1])

Y1 = Array{Float64}(undef, vs.nu, size(data_shakhov, 2))
for i in axes(Y1, 2)
    df = @view Y1[:, i]
    f = data_shakhov[:, i]
    q = heat_flux(f, prim0, vs.u, vs.weights)
    S = shakhov(vs.u, M0, q, prim0, 2 / 3)
    p = (M0 + S, τ0)
    bgk_ode!(df, f, p, 0.0)
end

nm = vs.nu
nν = vs.nu + 1

mnet = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, 3; bias = false))
νnet = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
#mnet1 = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm; bias = false))

u = [init_params(mnet1); init_params(νnet)]
γ = 3
ℓ = param_length(mnet1)
L = size(Y1, 2)

function loss(p, x, y)
    ℓ = param_length(mnet)
    #ℓ = param_length(mnet1)

    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = f_maxwellian(f, vs, γ)

    y = f .- M
    α = mnet(f, p[1:ℓ])
    #S = sigmoid(invariant_pdf(α, vs)) .* 2
    S = invariant_pdf(α, vs)

    #S = mnet1(y, p[1:ℓ])

    #α = mnet(f, p[1:ℓ])
    #S = sigmoid(invariant_pdf(α, vs)) .* 2
    #S = invariant_pdf(α, vs)

    z = vcat(y, τ)
    pred = ((M .* S) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(νnet(z, p[ℓ+1:end]))))
    #pred = ((M .* S) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(νnet(z, p[ℓ+1:end]))))
    #pred = ((S) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(νnet(z, p[ℓ+1:end]))))

    return sum(abs2, pred - y) / L
end

lold = loss(u, X1, Y1)

cb = function (p, l)
    println("loss: $(l)")
    return false
end
cb = throttle(cb, 1)

loss(p) = loss(p, X1, Y1)

res = sci_train(loss, u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 100)
res = sci_train(loss, res.u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)

lnew = loss(res.u)

#@save "newmodel.jld2" mnet νnet u X Y

loss(res.u)

function dfdt(df, f, p, t)
    mnet, νnet, u, vs, γ = p

    ℓ = param_length(mnet)
    M = f_maxwellian(f, vs, γ)

    y = f .- M
    α = mnet(y, u[1:ℓ])
    S = invariant_pdf(α, vs)

    z = vcat(y, τ0)
    df .= ((M .* S) .- f) ./ (τ0 .* (1 .+ 0.9 .* elu.(νnet(z, u[ℓ+1:end]))))
end

#df = zero(f0)
#dfdt(df, f0, (mnet, νnet, u, vs, γ), 0)

ube = ODEProblem(dfdt, f0, tspan, (mnet, νnet, res.u, vs, γ))
#ube = ODEProblem(dfdt, f0, tspan, (mnet, νnet, zero(u), vs, γ))
sol = solve(ube, Midpoint(); saveat = tsteps)

using Plots

idx = 5
begin
    plot(vs.u, sol.u[idx]; label = "nn")
    plot!(vs.u, data_bgk[:, idx]; label = "BGK", line = :dash)
    scatter!(vs.u, data_shakhov[:, idx]; label = "Shakhov", line = :dashdot)
end



function dfdt(f, p, t)
    df = zero(f)
    dfdt(df, f, p, t)
end

idx = 1
y = dfdt(data_bgk[:, idx], (mnet, νnet, res.u, vs, γ), 0)

begin
    plot(vs.u, y; label = "nn")
    plot!(vs.u, Y1[:, idx]; label = "BGK", line = :dash)
end