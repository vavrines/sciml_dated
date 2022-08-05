using KitBase, Solaris
using Solaris.Zygote
using Solaris.Flux, Solaris.Optimization
using ReverseDiff

vs = VSpace1D(-8, 8, 40; precision = Float32)
ut = zeros(Float32, vs.nu, 10)
wt = zeros(Float32, vs.nu, 10)


function f_maxwellian(g, f)
    w = [moments_conserve(g[:, i], vs.u, vs.weights) for i in axes(g, 2)]
    #prim = [conserve_prim(w1, 3) for w1 in w]
    #M = [maxwellian(vs.u, prim1) for prim1 in prim]
    #M1 = hcat(M...)

    return w[1][1] .* f
end

function f_maxwellian(f)
    #w = moments_conserve(f, vs.u, vs.weights)
    #prim = conserve_prim(w, 3)
    #return maxwellian(vs.u, prim)

    #ρ = sum(f .* vs.weights)
    #ρU = sum(f .* vs.u .* vs.weights)
    #ρE = 0.5 * sum(f .* vs.u.^2 .* vs.weights)

    #w = [moments_conserve(f[:, i], vs.u, vs.weights) for i in axes(f, 2)]
    #prim = [conserve_prim(w1, 3) for w1 in w]
    #M = [maxwellian(vs.u, prim1) for prim1 in prim]
    #M = [maxwellian(vs.u, w1) for w1 in w]
    #M1 = hcat(M...)

    #return M1
    g = Zygote.Buffer(f)
    return f_maxwellian(g, f)
end

struct BGKNet{T1,T2,T3}
    Mnet::T1
    νnet::T2
    fn::T3
end

BGKNet(m, ν) = BGKNet(m, ν, -)

(nn::BGKNet)(x, p) = begin
    np1 = param_length(nn.Mnet)
    M = f_maxwellian(x)

    nn.νnet(x, p[np1+1:end]) .* (nn.fn(M .+ nn.Mnet(x, p[1:np1]), x))
end

SR.init_params(nn::BGKNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh), FnDense(vs.nu * 2, vs.nu))
νn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh), FnDense(vs.nu * 2, vs.nu, sigmoid))

nn = BGKNet(mn, νn)
p = init_params(nn)

X = rand(Float32, vs.nu, 10)
Y = randn(Float32, vs.nu, 10)

#sci_train(nn, (X, Y), p, Solaris.Flux.Adam(), Optimization.AutoReverseDiff)
sci_train(nn, (X, Y), p)
