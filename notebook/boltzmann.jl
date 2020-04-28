using FFTW, OffsetArrays, FastGaussQuadrature, SpecialFunctions

get_kn_bzm(mu_ref, alpha) = 64 * sqrt(2.)^alpha / 5. * gamma((alpha+3) / 2) * gamma(2.) * sqrt(pi) * mu_ref

function lgwt(N,a,b)
    
    x = zeros(N)
    w = zeros(N)
    
    N1 = N
    N2 = N + 1
    
    y = zeros(N1); y0 = zeros(N1); Lp = zeros(N1)
    L = zeros(N1, N2)
    
    # initial guess
    for i=1:N1
        y[i] = cos( (2.0*(i-1.0)+1.0)*4.0*atan(1.0)/(2.0*(N-1.0)+2.0)  ) +
        0.27/N1*sin(4.0*atan(1.0)*(-1.0+i*2.0/(N1-1.0))*(N-1.0)/N2 )
        y0[i] = 2.
    end
    
    L .= 0.
    Lp .= 0.
    
    # compute the zeros of the N+1 legendre Polynomial
    # using the recursion relation and the Newton method
    while(maximum(abs.(y.-y0))>0.0000000000001)
        L[:,1].=1.0
        L[:,2].=y
        for k=2:N1
            @. L[:,k+1]=( (2.0*k-1.0)*y*L[:,k]-(k-1)*L[:,k-1] )/k
        end
        @. Lp=N2*( L[:,N1]-y*L[:,N2] )/(1.0-y*y)
        @. y0 = y
        @. y = y0 - L[:,N2]/Lp
    end
    
    # linear map from [-1 1] to [a,b]
    @. x=( a*(1.0-y)+b*(1.0+y) )/2.0
    @. w=N2*N2*(b-a)/( (1.0-y*y)*Lp*Lp ) /N1/N1

    return x, w
end 



function boltzmann_fft!( f::Array{<:Real,3}, Kn::Real, M::Int,
                         ϕ::Array{<:Real,4}, ψ::Array{<:Real,4}, phipsi::Array{<:Real,3} )

    #f_spec = OffsetArray{Complex{Float64}}(undef, axes(f, 1), axes(f, 2), axes(f, 3))
    f_spec = f .+ 0im
    bfft!(f_spec)
    f_spec ./= size(f, 1) * size(f, 2) * size(f, 3)
    f_spec .= fftshift(f_spec)

    #--- gain term ---#
    f_temp = zeros(axes(f_spec)) .+ 0im
    for i = 1:M*(M-1)
        fg1 = f_spec .* ϕ[:,:,:,i]
        fg2 = f_spec .* ψ[:,:,:,i]
        fg11 = fft(fg1)
        fg22 = fft(fg2)
        f_temp .+= fg11 .* fg22
    end

    #--- loss term ---#
    fl1 = f_spec .* phipsi
    fl2 = f_spec
    fl11 = fft(fl1)
    fl22 = fft(fl2)
    f_temp .-= fl11 .* fl22
    
    Q = @. 4. * π^2 / Kn / M^2 * real(f_temp)

    return Q

end

function kernel_mode( M::Int, umax::Real, vmax::Real, wmax::Real, du::Real, dv::Real, dw::Real, unum::Int, vnum::Int, wnum::Int, 
                      alpha::Real; quad_num=64 )

    supp = sqrt(2.) * 2. * max(umax, vmax, wmax) / (3. + sqrt(2.))

    fre_vx = range(-π / du, (unum÷2-1) * 2. * π / unum / du, length=unum)
    fre_vy = range(-π / dv, (vnum÷2-1) * 2. * π / vnum / dv, length=vnum)
    fre_vz = range(-π / dw, (wnum÷2-1) * 2. * π / wnum / dw, length=wnum)
    
    
    
    #abscissa, gweight = gausslegendre(quad_num)
    #@. abscissa = (0. * (1. - abscissa) + supp * (1. + abscissa)) / 2
    #@. gweight *= (supp - 0.) / 2
    
    abscissa, gweight = lgwt(quad_num, 0, supp)
    
    phi = zeros(unum, vnum, wnum, M*(M-1))
    psi = zeros(unum, vnum, wnum, M*(M-1))
    phipsi = zeros(unum, vnum, wnum)
    for loop = 1:M-1
        theta = π / M * loop
        for loop2 = 1:M
            theta2 = π / M * loop2
            idx = (loop - 1) * M + loop2
            for k in 1:wnum, j in 1:vnum, i in 1:unum
                s = fre_vx[i] * sin(theta) * cos(theta2) + 
                    fre_vy[j] * sin(theta) * sin(theta2) + 
                    fre_vz[k] * cos(theta)
                # phi
                int_temp = 0.
                for id in 1:quad_num
                    int_temp += 2. * gweight[id] * cos(s * abscissa[id]) * (abscissa[id]^alpha)
                end
                phi[i,j,k,idx] = int_temp * sin(theta)
                # psi
                s = fre_vx[i]^2 + fre_vy[j]^2 + fre_vz[k]^2 - s^2
                if s <= 0.
                    psi[i,j,k,idx] = π * supp^2
                else
                    s = sqrt(s)
                    bel = supp * s
                    bessel = besselj(1, bel)
                    psi[i,j,k,idx] = 2. * π * supp * bessel / s
                end
                # phipsi
                phipsi[i,j,k] += phi[i,j,k,idx]*psi[i,j,k,idx]
            end
        end
    end

    return phi, psi, phipsi

end