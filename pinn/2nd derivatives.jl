using KitBase, Flux, Optim, Optimization, Plots, Solaris
using Solaris.DiffEqFlux: FastDense, FastChain, initial_params
using Flux.Zygote: pullback

msplit(m) = begin
    mat_x(m), mat_y(m)
end
mat_x(m) = eltype(m).([1.0 0.0]) * m
mat_y(m) = eltype(m).([0.0 1.0]) * m

xspan = 0.f0:0.1f0:2.f0
yspan = 0.f0:0.1f0:1.f0

x = collect(xspan)
y = collect(yspan)
nx = length(x)
ny = length(y)
xMesh, yMesh = meshgrid(x, y)
xMesh1D = reshape(xMesh, (1, :))
yMesh1D = reshape(yMesh, (1, :))
mesh = cat(xMesh1D, yMesh1D; dims=1)

X = deepcopy(mesh)
Y = zeros(Float32, 1, length(x)*length(y));

m = FastChain(FastDense(2, 20, tanh), FastDense(20, 20, tanh), FastDense(20, 1))
p = initial_params(m);
θ = [vcat(X...); p];
u0 = vcat(X...)
n = length(Y);

u(x) = sin.(π .* mat_x(x) ./ 2.f0) .* mat_y(x) .+ 
    mat_x(x) .* (2.f0 .- mat_x(x)) .* mat_y(x) .* (1.f0 .- mat_y(x)) .* m(x, p);

u(x) =  m(x, p)

function loss(p)        
    u(x) = sin.(π .* mat_x(x) ./ 2.f0) .* mat_y(x) .+ 
        mat_x(x) .* (2.f0 .- mat_x(x)) .* mat_y(x) .* (1.f0 .- mat_y(x)) .* m(x, p);
    
    ux(x) = pullback(u, x)[2](ones(size(x)))[1] |> mat_x
    uy(x) = pullback(u, x)[2](ones(size(x)))[1] |> mat_y
    
    uxx(x) = pullback(ux, x)[2](ones(size(x)))[1] |> mat_x
    uyy(x) = pullback(uy, x)[2](ones(size(x)))[1] |> mat_y
 
    pred = uxx(X) + uyy(X)
    loss = sum(abs2, pred)
    
    return loss
end

cb = function (θ, l)
    display(l)
    return false
end

res = sci_train(loss, p, ADAM(), Optimization.AutoForwardDiff(); cb=cb, maxiters=20)


ux(x) = pullback(u, x)[2](ones(size(x)))[1] |> mat_x
uxx(x) = pullback(ux, x)[2](ones(size(x)))[1] |> mat_x

ux(X)
uxx(X)

