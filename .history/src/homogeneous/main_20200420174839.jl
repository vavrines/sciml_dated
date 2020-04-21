using Revise
using Kinetic

config = "config.txt"
println("Reading settings from $config \n")

# generate parameters
D = read_dict(config)
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

dim = ifelse(parse(Int, space[3]) >= 3, 3, parse(Int, space[1]))
γ = 1.4#heat_capacity_ratio(inK, dim)
vSpace = VSpace1D(u0, u1, nu, vMeshType, nug)

f0 = Float32.(0.3 * vSpace.u.^2 .* exp.(-0.3 .* vSpace.u.^2)) |> Array
w0 = [ discrete_moments(f0, vSpace.u, vSpace.weights, 0), 
       discrete_moments(f0, vSpace.u, vSpace.weights, 1), 
       discrete_moments(f0, vSpace.u, vSpace.weights, 2) ]
prim0 = conserve_prim(w0, γ)

M = Float32.(maxwellian(vSpace.u, prim0)) |> Array

f = similar(M)