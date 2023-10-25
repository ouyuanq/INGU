function TRD(J::Function; reltol=5e-15, inner_loops::Integer=20, display::Bool=false, mixed::Bool=true)
    # This function implements the trust-region-dogleg method to solve an approximate solution to nonlinear BVP F(u) = 0. The Newton step is solved approximately by GMRES and transpose of Jacobian is needed to compute the Cauchy point due to the usage of 2-norm.
    #  Input:  F = nonlinear BVP operator (multipledispatch of J)
    #          Jac = F' the corresponding Jacobian operator. J takes two inpus
    #              arguments u_k and v, where u_k is the current iteration and
    #              v is an arbitrary function. J_k(v) = J(u_k, v).
    #          Jact = the transpose of J (multipledispatch of J)
    #          reltol = the final relative tolerance of the residual
    #          inner_loops = maximum iterations of linear solver
    #          display = whether to show some key information during iterations
    #          mixed = whether mixed precision is used
    #
    #  Output: uCurrent = solution
    #          normFxOld = norm of residual of uCurrent 

    # key parameters
    kmax = 200  # maximum number of iterations
    eta1 = 0.25  # minimum acceptable ration of actual to predicted reduction
    eta2 = 0.75  # good ration of actual to predicted reduction
    Deltabar = 1e3 # The maximum size of trust-region
    omega_min, omega_max = 1e-5, 1e-1  # the smallest and largest tolerance of each GMRES iteration

    # initialization
    omega = 0.1  # the tolerance for each GMRES iteration for solving the Newton step
    uCurrent = J(Float64)
    delta_N = Vector{Float32}(undef, 0)  # the Newton step
    g = Vector{Float32}(undef, 0)  # the gradient
    Delta = 0.1  # the initial size of trust-region
    FCurrent = J(uCurrent)  # the function values of current iteration
    FTrial = similar(FCurrent)  # the function values of trial step
    norm2FxOld = sum(abs2, FCurrent)  # the norm of residual of the last iteration
    newNewton, newCauchy = true, true  # whether the inexact Newton step and the Cauchy point should be computed
    N = length(uCurrent)  # the maximum differential order
    Plan, Plani = plan_fft!(Vector{ComplexF32}(undef, 1)), plan_ifft!(Vector{ComplexF32}(undef, 1))

    # compute the final tolerance
    res_final = reltol^2 * norm2FxOld + reltol^2

    # Test if u0 is an appropriate solution already
    if norm2FxOld < res_final
        return uCurrent, sqrt(norm2FxOld)
    end

    # Newton iterations begin
    for k = 1:kmax
        if Delta < 1e-13
            # a stationary point is reached and no more progress can be made
            break
        end

        if display
            @printf "No.%i iteration \n" k
            @printf "Length of current iteration: %i \n" length(uCurrent)
            @printf "Residual: %.3e \n" sqrt(norm2FxOld)
            @printf "Relative tolerance for GMRES:%.3e \n" omega
            @printf "Radius of trust-region of No.%i iteration: %.3e \n \n" k Delta
        end

        # the coefficients of Jacobian at uCurrent
        if mixed
            JacCoeffs = J(convert(Vector{Float32}, uCurrent), 'C')
        else
            JacCoeffs = J(copy(uCurrent), 'C')  
        end

        # solve for the inexact Newton step
        if newNewton
            delta_N, Plan, Plani, _, _, _ = JacSolver!(J, FCurrent, JacCoeffs, omega, inner_loops)
            newNewton = false
        end

        # solve the trust-region subproblem approximately by dog-leg method
        normdN = norm(delta_N)
        if normdN < Delta
            # Newton step is within the region and we accept it
            delta = delta_N
        else
            # Cauchy point is needed: delta_C = -(||JᵀF||^2 / ||JJᵀF||^2) * JᵀF
            if newCauchy
                # pay attention to the function spaces that Jacobian and its transpose act on
                # Jᵗ : ultraspherical -> Chebyshev, F(u_k) is a ultraspherical series, g = JᵗF is the gradient
                g = J('T', JacCoeffs, view(FCurrent, 1:min(length(Plan), length(FCurrent))), Plan, Plani)

                # pay attention to the function spaces that Jacobian and its transpose act on
                # J : Chebyshev -> ultraspherical, g is a Chebyshev series, Jg is a C^{(lambda)} series
                Jg = J(similar(g), JacCoeffs, g, Plan, Plani)
                # invert the ultraspherical series to a Chebyshev series for consistency with the gradient g
                fastinve!(view(Jg, N+1:length(Jg)), N)

                # Cauchy point (with opposite direction)
                rmul!(g, sum(abs2, g) / sum(abs2, Jg))
                newCauchy = false
            end
            delta_C = copy(g)
            normdC = norm(delta_C)

            if normdC >= Delta
                # Cauchy point is out of the region. Take the largest step along gradient direction.
                delta = lmul!(-Delta / normdC, delta_C)
            else
                # from this point on we will only need delta_N in the term delta_N-delta_C,
                # so we reuse the vector delta_N by computing delta_N = delta_N - delta_C
                delta_diff = coeffsadd!(true, delta_C, copy(delta_N))
                rmul!(delta_C, -1)

                # Compute the optimal point on dogleg path
                # ||delta_C + tau*(delta_N - delta_C)||^2 = Delta^2
                b = 2 * myinnerproduct(delta_C, delta_diff)
                a = sum(abs2, delta_diff)
                c = normdC^2 - Delta^2

                tau = (-b + sqrt(b^2 - 4 * a * c)) / (2a)

                rmul!(delta_diff, tau)
                delta = coeffsadd!(true, delta_diff, delta_C)
            end
        end

        # take the trial step and compute the reduction ratio
        Jk_delta = J(similar(delta), JacCoeffs, delta, Plan, Plani)
        coeffsadd!(true, FCurrent, Jk_delta)
        predicted = sum(abs2, Jk_delta) # ||F_k + J_k delta_k||^2
        uTrial = coeffsadd_chop(true, delta, uCurrent)  # the coefficients of trial step
        # Compute the residual of trial step
        FTrial = J(uTrial)
        norm2FxTrial = sum(abs2, FTrial)

        # reduction ratio
        rho = (norm2FxOld - norm2FxTrial) / (norm2FxOld - predicted)

        # determine whether to accept the trial step according to the reduction ration
        if rho > 1e-1 && norm2FxOld > norm2FxTrial
            # step accepted
            if norm2FxTrial < res_final
                # we have found an approximate solution or reached a stationary point which is not a solution
                uCurrent = uTrial
                norm2FxOld = norm2FxTrial
                break
            end

            # new steps are needed
            newNewton, newCauchy = true, true

            # update the tolerance of GMRES
            # choice 2
            omega = 0.9 * norm2FxTrial / norm2FxOld

            # avoid the oversolving of final steps
            omega = max(omega, sqrt(res_final/(4*norm2FxTrial)))
            # ensure that omega ∈ [omega_min, omega_max]
            omega = max(omega_min, min(omega_max, omega))

            # update the iteration and the coefficients of Jacobian
            uCurrent = uTrial
            norm2FxOld = norm2FxTrial
            FCurrent = FTrial
        end

        if k > kmax
            # too many iterations
            break
        end

        # update size of trust-region
        nd = norm(delta)
        if rho < eta1 || norm2FxOld < norm2FxTrial
            # reduction of trust-region due to poor quadratic approximation
            Delta = nd / 4
        elseif rho >= eta2 && ≈(nd, Delta)
            # extension of trust-region due to well quadratic approximation
            Delta = min(Deltabar, 2 * Delta)
        end
    end

    if display
        @printf "Length of final solution: %i \n" length(uCurrent)
        @printf "Residual: %.3e \n" sqrt(norm2FxOld)
    end

    return uCurrent, sqrt(norm2FxOld)
end

function TRD_LU(J::Function; reltol=5e-15, display::Bool=false)
    # Same as TRD but replace the inner iterations (GMRES) by direct method (linsolve!)

    # key parameters
    kmax = 200  # maximum number of iterations
    eta1 = 0.25  # minimum acceptable ration of actual to predicted reduction
    eta2 = 0.75  # good ration of actual to predicted reduction
    Deltabar = 1e3 # The maximum size of trust-region
    omega_min, omega_max = 1e-5, 1e-1  # the smallest and largest tolerance of each GMRES iteration

    # initialization
    omega = 0.1  # the tolerance for each GMRES iteration for solving the Newton step
    uCurrent = J(Float64)
    delta_N = Vector{Float32}(undef, 0)  # the Newton step
    g = Vector{Float32}(undef, 0)  # the gradient
    Delta = 0.1  # the initial size of trust-region
    FCurrent = J(uCurrent)  # the function values of current iteration
    FTrial = similar(FCurrent)  # the function values of trial step
    norm2FxOld = sum(abs2, FCurrent)  # the norm of residual of the last iteration
    newNewton, newCauchy = true, true  # whether the inexact Newton step and the Cauchy point should be computed
    N = length(uCurrent)  # the maximum differential order
    L = zeros(Float64, 0, 0)

    # compute the final tolerance
    res_final = reltol^2 * norm2FxOld + reltol^2

    # Test if u0 is an appropriate solution already
    if norm2FxOld < res_final
        return uCurrent, sqrt(norm2FxOld)
    end

    # Newton iterations begin
    for k = 1:kmax
        if Delta < 1e-13
            # a stationary point is reached and no more progress can be made
            break
        end

        if display
            @printf "No.%i iteration \n" k
            @printf "Length of current iteration: %i \n" length(uCurrent)
            @printf "Residual: %.3e \n" sqrt(norm2FxOld)
            @printf "Relative tolerance for GMRES:%.3e \n" omega
            @printf "Radius of trust-region of No.%i iteration: %.3e \n \n" k Delta
        end

        # solve for the Newton step
        if newNewton
            delta_N, _, _, L = JacSolver_LU!(uCurrent, J, FCurrent, omega)
            newNewton = false
        end

        # solve the trust-region subproblem approximately by dog-leg method
        normdN = norm(delta_N)
        if normdN < Delta
            # Newton step is within the region and we accept it
            delta = delta_N
        else
            # Cauchy point is needed: delta_C = -(||JᵀF||^2 / ||JJᵀF||^2) * JᵀF
            if newCauchy
                # pay attention to the function spaces that Jacobian and its transpose act on
                # Jᵗ : ultraspherical -> Chebyshev, F(u_k) is a ultraspherical series, g = JᵗF is the gradient
                g = transpose(L) * copyto!(zeros(Float64, size(L, 1)), view(FCurrent, 1:min(size(L, 1), length(FCurrent))))

                # pay attention to the function spaces that Jacobian and its transpose act on
                # J : Chebyshev -> ultraspherical, g is a Chebyshev series, Jg is a C^{(lambda)} series
                Jg = L * g
                # invert the ultraspherical series to a Chebyshev series for consistency with the gradient g
                fastinve!(view(Jg, N+1:length(Jg)), N)

                # Cauchy point (with opposite direction)
                rmul!(g, sum(abs2, g) / sum(abs2, Jg))
                newCauchy = false
            end
            delta_C = copy(g)
            normdC = norm(delta_C)

            if normdC >= Delta
                # Cauchy point is out of the region. Take the largest step along gradient direction.
                delta = lmul!(-Delta / normdC, delta_C)
            else
                # from this point on we will only need delta_N in the term delta_N-delta_C,
                # so we reuse the vector delta_N by computing delta_N = delta_N - delta_C
                delta_diff = coeffsadd!(true, delta_C, copy(delta_N))
                rmul!(delta_C, -1)

                # Compute the optimal point on dogleg path
                # ||delta_C + tau*(delta_N - delta_C)||^2 = Delta^2
                b = 2 * myinnerproduct(delta_C, delta_diff)
                a = sum(abs2, delta_diff)
                c = normdC^2 - Delta^2

                tau = (-b + sqrt(b^2 - 4 * a * c)) / (2a)

                rmul!(delta_diff, tau)
                delta = coeffsadd!(true, delta_diff, delta_C)
            end
        end

        # take the trial step and compute the reduction ratio
        Jk_delta = L * copyto!(zeros(Float64, size(L, 1)), delta)
        coeffsadd!(true, FCurrent, Jk_delta)
        predicted = sum(abs2, Jk_delta) # ||F_k + J_k delta_k||^2
        uTrial = coeffsadd_chop(true, delta, uCurrent)  # the coefficients of trial step
        # Compute the residual of trial step
        FTrial = J(uTrial)
        norm2FxTrial = sum(abs2, FTrial)

        # reduction ratio
        rho = (norm2FxOld - norm2FxTrial) / (norm2FxOld - predicted)

        # determine whether to accept the trial step according to the reduction ration
        if rho > 1e-1 && norm2FxOld > norm2FxTrial
            # step accepted
            if norm2FxTrial < res_final
                # we have found an approximate solution or reached a stationary point which is not a solution
                uCurrent = uTrial
                norm2FxOld = norm2FxTrial
                break
            end

            # new steps are needed
            newNewton, newCauchy = true, true

            # update the tolerance of GMRES
            # choice 2
            omega = 0.9 * norm2FxTrial / norm2FxOld

            # avoid the oversolving of final steps
            omega = max(omega, sqrt(res_final/(4*norm2FxTrial)))
            # ensure that omega ∈ [omega_min, omega_max]
            omega = max(omega_min, min(omega_max, omega))

            # update the iteration and the coefficients of Jacobian
            uCurrent = uTrial
            norm2FxOld = norm2FxTrial
            FCurrent = FTrial
        end

        if k > kmax
            # too many iterations
            break
        end

        # update size of trust-region
        nd = norm(delta)
        if rho < eta1 || norm2FxOld < norm2FxTrial
            # reduction of trust-region due to poor quadratic approximation
            Delta = nd / 4
        elseif rho >= eta2 && ≈(nd, Delta)
            # extension of trust-region due to well quadratic approximation
            Delta = min(Deltabar, 2 * Delta)
        end
    end

    if display
        @printf "Length of final solution: %i \n" length(uCurrent)
        @printf "Residual: %.3e \n" sqrt(norm2FxOld)
    end

    return uCurrent, sqrt(norm2FxOld)
end