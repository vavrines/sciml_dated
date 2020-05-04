# ------------------------------------------------------------
# Normal shock structure
# ------------------------------------------------------------

using Revise
using DifferentialEquations
using Flux
using DiffEqFlux
using Optim
using Plots
using Dates
using FileIO
using JLD2

#using Kinetic
include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
using .Kinetic

function boltzmann!(df, f::Array{<:Real,3}, p, t)
    Kn, M, phi, psi, phipsi = p
    df .= boltzmann_fft(f, Kn, M, phi, psi, phipsi)
end

ks = SolverSet("shock.txt")

D = read_dict("nn.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

tSpan = (Float32(t0), Float32(t1))
tRan = range(tSpan[1], tSpan[2], length=tLen)
kn_bzm = hs_boltz_kn(ks.gas.μᵣ, ks.gas.Kn)

phi, psi, phipsi = kernel_mode(
	nm, ks.vSpace.u1, ks.vSpace.v1, ks.vSpace.w1,
	ks.vSpace.du[1,1,1], ks.vSpace.dv[1,1,1], ks.vSpace.dw[1,1,1],
	ks.vSpace.nu, ks.vSpace.nv, ks.vSpace.nw, ks.gas.αᵣ )

ML = Float32.(maxwellian(ks.vSpace.u, ks.vSpace.v, ks.vSpace.w, ks.ib.primL)) |> Array
MR = Float32.(maxwellian(ks.vSpace.u, ks.vSpace.v, ks.vSpace.w, ks.ib.primR)) |> Array

f1 = @. 0.1 * ML + 0.9 * MR
f2 = @. 0.5 * ML + 0.5 * MR
f3 = @. 0.9 * ML + 0.1 * MR

prob = ODEProblem(boltzmann!, f1, tSpan, [kn_bzm, nm, phi, psi, phipsi])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D1 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D1, 2)
    data_boltz_1D1[:,j] .= reduce_distribution(data_boltz[:,:,:,j], ks.vSpace.weights, 1)
end

prob = ODEProblem(boltzmann!, f2, tSpan, [kn_bzm, nm, phi, psi, phipsi])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D2 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D2, 2)
    data_boltz_1D2[:,j] .= reduce_distribution(data_boltz[:,:,:,j], ks.vSpace.weights, 1)
end

prob = ODEProblem(boltzmann!, f3, tSpan, [kn_bzm, nm, phi, psi, phipsi])
data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;
data_boltz_1D3 = zeros(Float32, axes(data_boltz, 1), axes(data_boltz, 4))
for j in axes(data_boltz_1D3, 2)
    data_boltz_1D3[:,j] .= reduce_distribution(data_boltz[:,:,:,j], ks.vSpace.weights, 1)
end

data_boltz_1D = [data_boltz_1D1, data_boltz_1D2, data_boltz_1D3]
f0_1D = [reduce_distribution(f1, ks.vSpace.weights, 1),
		 reduce_distribution(f2, ks.vSpace.weights, 1),
		 reduce_distribution(f3, ks.vSpace.weights, 1)]

plot(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[1][:,1], lw=2)
plot!(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[1][:,end], lw=2)
plot!(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[2][:,1], lw=2)
plot!(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[2][:,end], lw=2)
plot!(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[3][:,1], lw=2)
plot!(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[3][:,end], lw=2)

#--- neural ode ---#
dudt = FastChain( (x, p) -> x.^2,
                   FastDense(ks.vSpace.nu, ks.vSpace.nu*nh, tanh),
                   FastDense(ks.vSpace.nu*nh, ks.vSpace.nu) )
n_ode = NeuralODE(dudt, tSpan, Tsit5(), saveat=tRan)

function loss_n_ode(p)
    pred = [n_ode(f0_1D[1], p), n_ode(f0_1D[2], p), n_ode(f0_1D[3], p)]
    loss = sum(abs2, pred[1] .- data_boltz_1D[1]) +
		   sum(abs2, pred[2] .- data_boltz_1D[2]) +
		   sum(abs2, pred[3] .- data_boltz_1D[3])
    return loss, pred[1]
end

cb = function (p, l, pred; doplot=false)
    display(l)
    # plot current prediction against dataset
    if doplot
        pl = plot(tRan, data_boltz_1D[1][ks.vSpace.nu÷2,:], lw=2, label="Exact")
        scatter!(pl, tRan, pred[ks.vSpace.nu÷2,:], lw=2, label="NN prediction")
        display(plot(pl))
    end
    return false
end

res = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(), cb=cb, maxiters=300)

res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.9), cb=cb, maxiters=300)

res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, BFGS(), cb=cb, maxiters=300)

plot(tRan, data_boltz_1D[3][ks.vSpace.nu÷2,:], lw=2)


#plot(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], data_boltz_1D[3][:,:], lw=2, label="Dataset")
plot(ks.vSpace.u[:,ks.vSpace.nv÷2,ks.vSpace.nw÷2], n_ode(f0_1D[2], res.minimizer).u[:], lw=2, label="NN")


function loop!(ks, ctr, simTime)

	t = deepcopy(simTime)
	iter = 0
	dt = 0.
	res = zeros(axes(ks.ib.wL))
	dt = Kinetic.timestep(ks, ctr, simTime)

	while simTime < ks.set.maxTime

		Kinetic.evolve!(ks, ctr, face, dt)

		Kinetic.update!(ks, ctr, face, dt, res)

		iter += 1
		t += dt

		if iter%10 == 0
			println("iter: $(iter), time: $(simTime), dt: $(dt), res: $(res[1:end])")
		end

		if maximum(res) < 5.e-7 || simTime > ks.set.maxTime
			break
		end

	end # while loop

	return t
end


ks, ctr, face, simTime = Kinetic.initialize("shock.txt")

#Kinetic.solve!( ks, ctr, face, simTime )

#plot_line(ks, ctr)
