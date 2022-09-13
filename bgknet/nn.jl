using Solaris
using Flux: elu, relu

act(x) = ifelse(x > 10, 10, x)

(nn::BGKNet)(x, p, vs, γ, ::Type{VDF{1,1}}, ::Type{Class{2}}) = begin
    f = @view x[begin:end-1, :]
    τ = @view x[end:end, :]
    M = f_maxwellian(f, vs, γ)
    y = f .- M
    z = vcat(y, τ)

    nm = param_length(nn.Mnet)
    α = nn.Mnet(y, p[1:nm])
    S = collision_invariant(α, vs)

    return (relu(M .* S) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end]))))
end

(nn::BGKNet)(x, p, vs, γ, ::Type{VDF{2,1}}, ::Type{Class{2}}) = begin
    f = @view x[begin:end-1, :]
    h = @view x[begin:vs.nu, :]
    b = @view x[vs.nu+1:end-1, :]
    τ = @view x[end:end, :]

    H, B = f_maxwellian(h, b, vs, γ)
    M = [H; B]
    y = f .- M
    z = vcat(y, τ)

    nm = param_length(nn.Mnet)
    α = nn.Mnet(y, p[1:nm])
    SH = collision_invariant(α[1:3], vs)
    SB = collision_invariant(α[4:end], vs)
    S = vcat(SH, SB)

    return (relu(M .* S) .- f) ./ (τ .* (1 .+ 0.9 .* elu.(nn.νnet(z, p[nm+1:end]))))
end
