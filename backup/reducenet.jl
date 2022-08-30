using KitBase: AV, AM
using Solaris.Flux: relu, elu

struct BGKReduceNet{T1,T2}
    Mnet::T1
    νnet::T2
end

(nn::BGKReduceNet)(x, p, vs::VSpace1D = VSpace1D(-6, 6, (size(x)[1] - 1)÷2; precision = Float32), γ = 5/3) = begin
    nm = param_length(nn.Mnet)

    h = @view x[1:vs.nu, :]
    b = @view x[vs.nu+1:end-1, :]
    f = @view x[1:end-1, :]
    τ = @view x[end:end, :]

    H, B = f_maxwellian(h, b, vs, γ)
    M = [H; B]
    y = f .- M
  
    dM = nn.Mnet(y, p[1:nm])
    z = vcat(y, τ)
    
    (relu(M .+ dM) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end]))))
end

Solaris.init_params(nn::BGKReduceNet) = vcat(init_params(nn.Mnet), init_params(nn.νnet))
