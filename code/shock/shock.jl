# ------------------------------------------------------------
# Normal shock structure
# ------------------------------------------------------------

using Revise
using DifferentialEquations
using Flux
using DiffEqFlux
using Optim
using Plots
using FileIO
using JLD2
using OffsetArrays

#using Kinetic
include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
using .Kinetic

function bgk!(df, f, p, t)
    g, tau = p
    df .= (g .- f) ./ tau
end

D = read_dict("shock1D.txt")
for key in keys(D)
    s = Symbol(key)
    @eval $s = $(D[key])
end

γ = 3.
set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)
μᵣ = ref_vhs_vis(knudsen, alphaRef, omegaRef)
gas = GasProperty(knudsen, mach, prandtl, inK, γ, omega, alphaRef, omegaRef, μᵣ)
vSpace = VSpace1D(umin, umax, nu, vMeshType, nug)
wL, primL, fL, bcL, wR, primR, fR, bcR = ib_rh(mach, γ, vSpace.u)
ib = IB1D1F(wL, primL, fL, bcL, wR, primR, fR, bcR)

ks = SolverSet(set, pSpace, vSpace, gas, ib, pwd())

ctr = OffsetArray{ControlVolume1D1F}(undef, eachindex(ks.pSpace.x))
face = Array{Interface1D1F}(undef, ks.pSpace.nx+1)
for i in eachindex(ctr)
    if i <= ks.pSpace.nx÷2
        ctr[i] = ControlVolume1D1F(ks.pSpace.x[i], ks.pSpace.dx[i], ks.ib.wL, ks.ib.primL, ks.ib.fL)
    else
        ctr[i] = ControlVolume1D1F(ks.pSpace.x[i], ks.pSpace.dx[i], ks.ib.wR, ks.ib.primR, ks.ib.fR)
    end
end
for i=1:ks.pSpace.nx+1
    face[i] = Interface1D1F(ks.ib.fL)
end

sos = sound_speed(ks.ib.primR, γ)
vmax = ks.vSpace.u1 + sos
tmax = vmax / ks.pSpace.dx[1]
dt = Float32(ks.set.cfl / tmax)
tSpan = (0.f0, dt)
tRan = range(tSpan[1], tSpan[2], length=tLen)

# stable structure
residual = Array{Float32}(undef, 3)
for iter=1:1000
    Kinetic.evolve!(ks, ctr, face, dt)
    Kinetic.update!(ks, ctr, face, dt, residual)
end

plot_line(ks, ctr)



cb = function (p, l, pred)
    display(l)
    return false
end


function step_train!( fwL, ffL, w, prim, f, fwR, ffR,
			          γ, μ, ω, u, n_ode, p, dx, tRan, RES, AVG )

    w_old = deepcopy(w)

    #--- update W^{n+1} ---#
	@. w += (fwL - fwR) / dx
	prim .= conserve_prim(w, γ)

	#--- record residuals ---#
	@. RES += (w - w_old)^2
	@. AVG += abs(w)

    M = maxwellian(u, prim)
	τ = vhs_collision_time(prim, μ, ω)
    prob = ODEProblem(bgk!, f, tSpan, [M, τ])
    data_boltz = solve(prob, Tsit5(), saveat=tRan) |> Array;

    function loss_n_ode(p)
        pred = n_ode(f[1:end], p)
        loss = sum(abs2, pred .- data_boltz)
        return loss, pred
    end

    if loss_n_ode(p)[1] <= 1.e-6
        ftemp = n_ode(Float32.(f[1:end]), p).u[end]
    	for i in eachindex(u)
    		f[i] = ftemp[i] + (ffL[i] - ffR[i]) / dx
    	end
    else
        optim = DiffEqFlux.sciml_train(loss_n_ode, p, ADAM(), cb=cb, maxiters=100)
        #optim = DiffEqFlux.sciml_train(loss_n_ode, p, LBFGS(), cb=cb, maxiters=400)
        p .= optim.minimizer

        for i in eachindex(u)
    		f[i] += (ffL[i] - ffR[i]) / dx + (M[i] - f[i]) / τ * tRan[end]
    	end
    end

end

function step_node!( fwL, ffL, w, prim, f, fwR, ffR,
			         γ, u, n_ode, p, dx, RES, AVG )

    w_old = deepcopy(w)

    #--- update W^{n+1} ---#
	@. w += (fwL - fwR) / dx
	prim .= conserve_prim(w, γ)

	#--- record residuals ---#
	@. RES += (w - w_old)^2
	@. AVG += abs(w)

	#--- update distribution function ---#
    ftemp = n_ode(Float32.(f[1:end]), p).u[end]
	for i in eachindex(u)
		f[i] = ftemp[i] + (ffL[i] - ffR[i]) / dx
	end

end

sumRes = zeros(Float32, axes(ib.wL))
sumAvg = zeros(Float32, axes(ib.wL))
for iter = 1:100

    Kinetic.evolve!(ks, ctr, face, dt)

    for i in 2:pSpace.nx-1
        step_train!( face[i].fw, face[i].ff, ctr[i].w, ctr[i].prim, ctr[i].f,
                     face[i+1].fw, face[i+1].ff, ks.gas.γ, ks.gas.μᵣ, ks.gas.ω, vSpace.u, n_ode, res.minimizer,
                     ctr[i].dx, tRan, sumRes, sumAvg)
    end

end

plot_line(ks, ctr)



pred = n_ode(ctr[1].f[1:end], res.minimizer)



sumRes = zeros(Float32, axes(ib.wL))
sumAvg = zeros(Float32, axes(ib.wL))
for i=2:ks.pSpace.nx-1
	Kinetic.step!( face[i].fw, face[i].ff, ctr[i].w, ctr[i].prim, ctr[i].f,
				   face[i+1].fw, face[i+1].ff, ks.gas.γ, ks.vSpace.u, ks.gas.μᵣ, ks.gas.ω,
				   ctr[i].dx, dt, sumRes, sumAvg )
end



#--- neural ode ---#
dudt = FastChain( (x, p) -> x.^3,
                   FastDense(vSpace.nu, vSpace.nu*12, tanh),
                   #FastDense(vSpace.nu*10, vSpace.nu*10, tanh),
                   FastDense(vSpace.nu*12, vSpace.nu) )
n_ode = NeuralODE(dudt, tSpan, Tsit5(), saveat=tRan)
@load "optimizer.jld2" res


n_ode(ctr[1].f[1:end], res.minimizer)




step( face[2].fw, face[2].ff, ctr[2].w, ctr[2].prim, ctr[2].f,
      face[3].fw, face[3].ff, ks.gas.γ, vSpace.u, n_ode, res.minimizer,
      ctr[2].dx, dt, sumRes, sumAvg)


for iter = 1:1

    Kinetic.evolve!(ks, ctr, face, dt)

    for i in 2:pSpace.nx-1
        ctr[i].w, ctr[i].prim, ctr[i].f =
        step( face[i].fw, face[i].ff, ctr[i].w, ctr[i].prim, ctr[i].f,
              face[i+1].fw, face[i+1].ff, ks.gas.γ, vSpace.u, n_ode, res.minimizer,
              ctr[i].dx, dt, sumRes, sumAvg)
    end

end



plot_line(ks, ctr)



for i in 2:pSpace.nx
    fw[i,:], ff[i,:] = flux_kfvs( fField[i-1,:], fField[i,:], Float32.(vSpace.u), Float32.(vSpace.weights), dt )
end

sumRes = zeros(Float32, axes(ib.wL))
sumAvg = zeros(Float32, axes(ib.wL))
for i in 2:pSpace.nx-1

    wField[i,:], primField[i,:], fField[i,:] = step( fw[i,:], ff[i,:], wField[i,:], primField[i,:], fField[i,:],
           fw[i+1,:], ff[i+1,:], gas.γ, vSpace.u, n_ode, res.minimizer,
           pSpace.dx[i], dt, sumRes, sumAvg )
#=
    w_old = deepcopy(wField[i,:])

	wField[i,:] .+= (fw[i,:] - fw[i+1,:]) / pSpace.dx[i]
	primField[i,:] .= conserve_prim(wField[i,:], γ)

    ftemp = n_ode(Float32.(fField[i,1:end]), res.minimizer).u[end]
	@. fField[i,:] = ftemp + (ff[i,:] - ff[i+1,:]) / pSpace.dx[i]
=#
end

draw = function()
    pltx = deepcopy(pSpace.x)
    plty = zeros(pSpace.nx, 6)
    for i in eachindex(pltx)
        for j=1:2
            plty[i,j] = primField[i,j]
        end

        plty[i,3] = 1. / primField[i,end]
    end
    p1 = plot(pltx, plty[:,1], label="Density", lw=2, xlabel="X")
    p1 = plot!(pltx, plty[:,2], label="Velocity", lw=2)
    p1 = plot!(pltx, plty[:,3], label="Temperature", lw=2)
    display(p1)
end

draw()

plot(vSpace.u[1:end], fField[2, 1:end], lw=2)

plot(vSpace.u, n_ode(Float32.(ib.fL[3:end]), res.minimizer).u[end], lw=2, label="Dataset")
