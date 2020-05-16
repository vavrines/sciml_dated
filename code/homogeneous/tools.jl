using DiffEqFlux

struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))

function vis_train(
    loss,
    _θ,
    opt,
    _data = DEFAULT_DATA;
    cb = (args...) -> false,
    maxiters = get_maxiters(data),
    progress = true,
    save_best = true,
)

    # Flux is silly and doesn't have an abstract type on its optimizers, so assume
    # this is a Flux optimizer
    θ = copy(_θ)
    ps = Flux.params(θ)

    if _data == DEFAULT_DATA && maxiters == typemax(Int)
        error("For Flux optimizers, either a data iterator must be provided or the `maxiters` keyword argument must be set.")
    elseif _data == DEFAULT_DATA && maxiters != typemax(Int)
        data = Iterators.repeated((), maxiters)
    elseif maxiters != typemax(Int)
        data = take(_data, maxiters)
    else
        data = _data
    end

    t0 = time()

    local x, min_err
    min_err = typemax(eltype(_θ)) #dummy variables
    min_opt = 1


    anim = @animate for (i, d) in enumerate(data)
        gs = Flux.Zygote.gradient(ps) do
            x = loss(θ, d...)
            first(x)
        end
        cb_call = cb(θ, x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
            break
        end
        msg = @sprintf("loss: %.3g", x[1])
        #progress && ProgressLogging.@logprogress msg i/maxiters
        DiffEqFlux.update!(opt, ps, gs)

        if save_best
            if first(x) < first(min_err)  #found a better solution
                min_opt = opt
                min_err = x
            end
            if i == maxiters  #Last iteration, revert to best.
                opt = min_opt
                cb(θ, min_err...)
            end
        end
    end

    _time = time()

    res = Optim.MultivariateOptimizationResults(
        opt,
        _θ,# initial_x,
        θ, #pick_best_x(f_incr_pick, state),
        first(x), # pick_best_f(f_incr_pick, state, d),
        maxiters, #iteration,
        maxiters >= maxiters, #iteration == options.iterations,
        false, # x_converged,
        0.0,#T(options.x_tol),
        0.0,#T(options.x_tol),
        NaN,# x_abschange(state),
        NaN,# x_abschange(state),
        false,# f_converged,
        0.0,#T(options.f_tol),
        0.0,#T(options.f_tol),
        NaN,#f_abschange(d, state),
        NaN,#f_abschange(d, state),
        false,#g_converged,
        0.0,#T(options.g_tol),
        NaN,#g_residual(d),
        false, #f_increased,
        nothing,
        maxiters,
        maxiters,
        0,
        true,
        NaN,
        _time - t0,
    )

    return res, anim
end
