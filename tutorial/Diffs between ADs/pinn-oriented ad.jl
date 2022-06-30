using KitBase, Flux, Optimization, Solaris
using ForwardDiff, ReverseDiff, Flux.Zygote
using Solaris.DiffEqFlux: FastDense, FastChain, initial_params

mat_x(m) = eltype(m).([1.0 0.0]) * m
mat_y(m) = eltype(m).([0.0 1.0]) * m

nn = FastChain(FastDense(2, 20, tanh), FastDense(20, 1))
p = initial_params(nn)

x = 0.f0:0.1f0:1.f0 |> collect
y = 0.f0:0.1f0:0.5f0 |> collect
xMesh, yMesh = meshgrid(x, y)
xMesh1D = reshape(xMesh, (1, :))
yMesh1D = reshape(yMesh, (1, :))
mesh = cat(xMesh1D, yMesh1D; dims=1)

X = deepcopy(mesh)
Y = zeros(Float32, 1, length(x)*length(y))

#u(x) = nn(x, p)
u(x) = mat_x(x) .* mat_y(x) .* nn(x, p)

# Zygote
ux1(x) = Zygote.pullback(u, x)[2](ones(size(x)))[1]# |> mat_x
ux1(X)
ux1(X[:, 2])

# ForwardDiff & ReverseDiff
ForwardDiff.jacobian(u, X[:, 2])
ReverseDiff.jacobian(u, X[:, 2])
Zygote.jacobian(u, X[:, 2])

# 结果差1个2X的系数？