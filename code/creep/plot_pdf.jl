using Kinetic, PyCall
using JLD2

begin
    set = Setup(
        "gas", # matter
        "creep", # case
        "2d2f2v", # space
        "kfvs", # flux
        "shakhov", # collision: for scalar conservation laws there are none
        1, # species
        2, # interpolation order
        "vanleer", # limiter
        "maxwell", # boundary
        0.8, # cfl
        100.0, # simulation time
    )

    ps = PSpace2D(0.0, 5.0, 200, 0.0, 1.0, 40)
    vs = VSpace2D(-5.0, 5.0, 28, -5.0, 5.0, 28)
    Kn = 0.064
    gas = KitBase.Gas(Kn, 0.0, 2/3, 1, 5/3, 0.81, 1.0, 0.5, ref_vhs_vis(Kn, 1.0, 0.5))

    prim0 = [1.0, 0.0, 0.0, 1.0]
    w0 = prim_conserve(prim0, 5/3)
    h0 = maxwellian(vs.u, vs.v, prim0)
    b0 = @. h0 * gas.K / (2 * prim0[end])
    bcL = deepcopy(prim0)
    bcR = [1.0, 0.0, 0.0, 273/573]
    ib = IB2F(w0, prim0, h0, b0, bcL, w0, prim0, h0, b0, bcR)

    ks = SolverSet(set, ps, vs, gas, ib, @__DIR__)
    cd(@__DIR__)
    @load "transition/ctr.jld2"
end

u = zeros(ks.vSpace.nu, ks.vSpace.nv, ks.pSpace.nx)
v = zero(u)
x = zero(u)
for i in axes(u, 1), j in axes(u, 2), k in axes(u, 3)
    u[i, :, :] .= ks.vSpace.u[i, 1]
    v[:, j, :] .= ks.vSpace.v[1, j]
    x[:, :, k] .= ks.pSpace.x[k, 1]
end
pdf = zeros(ks.vSpace.nu, ks.vSpace.nv, ks.pSpace.nx)
for k in axes(pdf, 3)
    pdf[:, :, k] .= (ctr[k, 20].h .+ ctr[k, 21].h) ./ 2
end

plotly = pyimport("plotly")
go = plotly.graph_objects
fig = go.Figure()
fig.add_trace(
    go.Volume(
        x=u[:],
        y=v[:],
        z=x[:],
        value=pdf[:],
        opacity=0.1,
        surface_count=31, 
    )
)
fig.update_layout(
    scene = Dict(
        "xaxis_title"=>"u",
        "yaxis_title"=>"v",
        "zaxis_title"=>"x"
    ),
)
fig.show()
fig.write_image("creep_pdf1.pdf")
fig.write_image("creep_pdf2.pdf")