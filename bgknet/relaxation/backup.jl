begin
    tspan = (0, set.maxTime)
    tsteps = linspace(tspan[1], tspan[2], set.tnum) .|> Float32
    γ = heat_capacity_ratio(set.K, 3)
    vs = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

    f0 =
        Float32.(
            0.5 * (1 / π)^1.5 .*
            (exp.(-(vs.u .- 0.99) .^ 2) .+ exp.(-(vs.u .+ 0.99) .^ 2)) .*
            exp.(-vs.v .^ 2) .* exp.(-vs.w .^ 2),
        ) |> Array
    prim0 =
        conserve_prim(moments_conserve(f0, vs.u, vs.v, vs.w, vs.weights), γ)
    M0 = Float32.(maxwellian(vs.u, vs.v, vs.w, prim0)) |> Array

    mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
    kn_bzm = hs_boltz_kn(mu_ref, 1.0)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    phi, psi, phipsi = kernel_mode(
        set.nm,
        vs.u1,
        vs.v1,
        vs.w1,
        vs.du[1, 1, 1],
        vs.dv[1, 1, 1],
        vs.dw[1, 1, 1],
        vs.nu,
        vs.nv,
        vs.nw,
        set.alpha,
    )

    # Boltzmann
    prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, phipsi])
    data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

    # BGK
    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    data_boltz_1D = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
    data_bgk_1D = zeros(Float32, axes(data_bgk, 1), axes(data_bgk, 4))
    for j in axes(data_boltz_1D, 2)
        data_boltz_1D[:, j] .=
            reduce_distribution(data_boltz[:, :, :, j], vs.weights[1, :, :])
        data_bgk_1D[:, j] .=
            reduce_distribution(data_bgk[:, :, :, j], vs.weights[1, :, :])
    end
    f0_1D = reduce_distribution(f0, vs.weights[1, :, :])
    M0_1D = reduce_distribution(M0, vs.weights[1, :, :])
end

rhs3D = zeros(Float32, vs.nu, vs.nv, vs.nw, size(X, 2))
for i in axes(rhs3D, 4)
    df = @view rhs3D[:, :, :, i]
    boltzmann_ode!(df, data_boltz[:, :, :, i], [kn_bzm, set.nm, phi, psi, phipsi], 0.0)
end

Y = Array{Float32}(undef, vs.nu, set.tnum)
for j in axes(rhs3D, 4)
    Y[:, j] .= reduce_distribution(rhs3D[:, :, :, j], vs.weights[1, :, :])
end

# 2nd
begin
    f0 =
        Float32.(
            0.8 * (1 / π)^1.5 .*
            (exp.(-(vs.u .- 0.8) .^ 2) .+ exp.(-(vs.u .+ 0.8) .^ 2)) .*
            exp.(-vs.v .^ 2) .* exp.(-vs.w .^ 2),
        ) |> Array
    prim0 =
        conserve_prim(moments_conserve(f0, vs.u, vs.v, vs.w, vs.weights), γ)
    M0 = Float32.(maxwellian(vs.u, vs.v, vs.w, prim0)) |> Array

    mu_ref = ref_vhs_vis(set.Kn, set.alpha, set.omega)
    kn_bzm = hs_boltz_kn(mu_ref, 1.0)
    τ0 = mu_ref * 2.0 * prim0[end]^(0.5) / prim0[1]

    # Boltzmann
    prob = ODEProblem(boltzmann_ode!, f0, tspan, [kn_bzm, set.nm, phi, psi, phipsi])
    data_boltz = solve(prob, Tsit5(), saveat = tsteps) |> Array

    # BGK
    prob1 = ODEProblem(bgk_ode!, f0, tspan, [M0, τ0])
    data_bgk = solve(prob1, Tsit5(), saveat = tsteps) |> Array

    data_boltz_1D = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
    data_bgk_1D = zeros(Float32, axes(data_bgk, 1), axes(data_bgk, 4))
    for j in axes(data_boltz_1D, 2)
        data_boltz_1D[:, j] .=
            reduce_distribution(data_boltz[:, :, :, j], vs.weights[1, :, :])
        data_bgk_1D[:, j] .=
            reduce_distribution(data_bgk[:, :, :, j], vs.weights[1, :, :])
    end
    f0_1D = reduce_distribution(f0, vs.weights[1, :, :])
    M0_1D = reduce_distribution(M0, vs.weights[1, :, :])
end

X1 = Array{Float32}(undef, vs.nu, size(data_boltz_1D, 2))
for i in axes(X1, 2)
    X1[:, i] .= data_boltz_1D[:, i]
end
M = Array{Float32}(undef, set.nu, size(X1, 2))
for i in axes(M, 2)
    M[:, i] .= M0_1D
end
τ = Array{Float32}(undef, 1, size(X1, 2))
for i in axes(τ, 2)
    τ[1, i] = τ0
end
X1 = vcat(X1, ones(Float32, 1, size(X1, 2)) .* τ[1])

rhs3D = zeros(Float32, vs.nu, vs.nv, vs.nw, size(X1, 2))
for i in axes(rhs3D, 4)
    df = @view rhs3D[:, :, :, i]
    boltzmann_ode!(df, data_boltz[:, :, :, i], [kn_bzm, set.nm, phi, psi, phipsi], 0.0)
end

Y1 = Array{Float32}(undef, vs.nu, set.tnum)
for j in axes(rhs3D, 4)
    Y1[:, j] .= reduce_distribution(rhs3D[:, :, :, j], vs.weights[1, :, :])
end

X = hcat(X, X1)
Y = hcat(Y, Y1)

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu + 1, vs.nu * 2 + 1, tanh; bias = false), FnDense(vs.nu * 2 + 1, vs.nu, sigmoid; bias = false))

nn = BGKNet(mn, νn)
p = init_params(nn)

vs1d = VSpace1D(set.u0, set.u1, set.nu; precision = Float32)

data = (X, Y)
L = size(data[1], 2)
loss(p) = sum(abs2, nn(data[1], p, vs1d) - data[2]) / L

his = []
cb = function (p, l)
    println("loss: $(loss(p))")
    push!(his, l)
    return false
end

res = sci_train(loss, p, Adam(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 5000)
res = sci_train(loss, res.u, LBFGS(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 1000)

cd(@__DIR__)
u = res.u
@save "minimizer.jld2" u
