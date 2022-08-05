using Flux, Solaris

struct BGKNet{T,T1,T2}
    chain::T
    f::T1
    σ::T2
end

BGKNet(chain) = BGKNet(chain, +, tanh)#BGKNet{typeof(chain),typeof(+),typeof(tanh)}(chain, +, tanh)

(nn::BGKNet)(x, p) = begin
    nn.σ.(nn.f(nn.chain(x, p), x))
end

m = FnChain(FnDense(4, 8, tanh), FnDense(8, 4))
nn = BGKNet(m)
p = init_params(nn.chain)

X = randn(Float32, 4, 10)

nn(X, p)

sci_train()
