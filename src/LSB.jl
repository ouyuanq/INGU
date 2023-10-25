function LSB(J::Function; reltol=5e-15, inner_loops::Integer=20, display::Bool=false, mixed::Bool=true)
    # Solve the nonlinear BVP F(x) = 0 by inexact Newton method where
    # F'(x_k)dx = - F(x_k) is solved approximately and the steplength is
    # reduced if the current Newton update doesn't decrease the residual.
    # GMRES is used in which the fast methods can be used.

    #  Input:  F = nonlinear BVP operator (multipledispatch of J)
    #          J = F' the corresponding Jacobian operator. J takes two inputs
    #              arguments u_k and v, where u_k is the current iteration and
    #              v is an arbitrary function. J_k(v) = J(u_k, v).
    #          reltol = the final relative tolerance of the residual
    #          inner_loops = maximum iterations of linear solver
    #          display = whether to show some key information during iterations
    #          mixed = whether mixed precision is used
    #
    #  Output: uCurrent = solution
    #          normFxOld = norm of residual of uCurrent 

    # key parameters
    omega_min, omega_max = 1e-5, 1e-1  # the smallest and largest tolerance of each GMRES iteration
    maxiter = 300  # maximum number of iterations
    bstep = 10  # maximum number of steps in backtracking

    # initialization
    uCurrent = J(Float64) # get the initial iteration
    omega = 0.01  # denotes the tolerance during GMRES iterations
    gamma = 0.9  # parameter of choice 2 forcing terms
    FCurrent = J(uCurrent)  # the function values of current iteration
    normFxOld = norm(FCurrent)  # the norm of residual of the last iteration
    normFxTrial = .0  # the norm of trial step, i.e., ||F(x_k) + F'(x_k)s_k||
    t = 1e-4  # restricted residual monotonicity parameter
    theta_min, theta_max = 0.1, 0.5  # the interval of backtracking steps

    # compute the final tolerance
    res_final = reltol * normFxOld + reltol  # relative tolerance and absolute tolerance

    if normFxOld < res_final
        # initial iteration is accurate enough
        return uCurrent, normFxOld
    end

    # Newton iterations begin
    for k = 1:maxiter
        if display
            @printf "No.%i iteration \n" k
            @printf "Length of current iteration: %i \n" length(uCurrent)
            @printf "Residual: %.3e \n" normFxOld
            @printf "Relative tolerance for GMRES:%.3e \n" omega
        end

        if mixed
            # the coefficients of Jacobian at uCurrent
            JacCoeffs = J(convert(Vector{Float32}, uCurrent), 'C')
        else
            JacCoeffs = J(copy(uCurrent), 'C')  
        end

        # solve for the inexact Newton step
        delta, Plan, Plani, _, _, _ = JacSolver!(J, FCurrent, JacCoeffs, omega, inner_loops)

        # choose the step by backtracking
        reduced = false
        g0 = normFxOld  # g(gamma) = ||F(uCurrent + gamma * delta)||, note that the norm is not squred
        omega1 = copy(omega)
        theta1 = 1.0
        Jk_delta = J(similar(delta), JacCoeffs, delta, Plan, Plani)
        gp0 = myinnerproduct(FCurrent, Jk_delta) / (-g0) # p'(0) where p is local quadratic model
        # take a trial step
        uTrial = coeffsadd_chop(true, delta, uCurrent)  # the coefficients of trial step
        # Compute the residual of trial step
        FTrial = J(uTrial)
        for i = 1:bstep
            g1 = norm(FTrial)
            if g1 < (1 - t * (1 - omega1)) * g0
                # the restricted residual monotonicity has been passed
                normFxTrial = g1
                reduced = true
                break
            else
                # choose parameter in backtracking interval using quadratic model
                # the coefficient of quadratic term
                qcoeff = g1 - g0 - gp0
                # axis of symmetry of quadratic model
                z = -gp0 / (2 * qcoeff)
                if qcoeff <= 0
                    # the direction is not a descent direction and the minimum value is at the endpoints
                    theta = theta_max
                else
                    # the minimum value is at the axis of symmetry or endpoints
                    if z > theta_max
                        theta = theta_max
                    elseif z < theta_min
                        theta = theta_min
                    else
                        theta = z
                    end
                end
            end

            # update
            rmul!(delta, theta) # backtracking
            rmul!(Jk_delta, theta) # Jacobian is linear
            gp0 *= theta # inner product is linear
            theta1 *= theta
            omega1 = 1 - theta * (1 - omega1)
            uTrial = coeffsadd_chop(true, delta, uCurrent)  # the coefficients of trial step
            FTrial = J(uTrial)
        end

        if !reduced
            # failure of backtracking and we return what we have gotten
            break
        end

        if display
            @printf "Steplength of No.%i iteration: %.3e \n \n" k theta1
        end

        if normFxTrial < res_final
            # the solution has been found
            uCurrent = uTrial
            normFxOld = normFxTrial
            break
        end

        # choose the forcing term of the next iteration
        # safeguard
        omega_safe = gamma * omega^2
        # choice 2
        omega = gamma * (normFxTrial / normFxOld)^2

        if omega_safe > 0.1
            # safeguard
            omega = max(omega, omega_safe)
        end

        # avoid the oversolving of final steps
        omega = max(omega, res_final/(2*normFxTrial))

        # ensure that omega ∈ [omega_min, omega_max]
        omega = max(omega_min, min(omega_max, omega))

        # we accept the backtracking step and update
        uCurrent = uTrial
        FCurrent = FTrial
        normFxOld = normFxTrial
        k = k + 1
    end

    if display
        @printf "Length of final solution: %i \n" length(uCurrent)
        @printf "Residual: %.3e \n" normFxOld
    end

    return uCurrent, normFxOld
end

function LSB_LU(J::Function; reltol=5e-15, display::Bool=false)
    # Same as LSB but replace the inner iterations (GMRES) by direct method (linsolve!)

    # key parameters
    omega_min, omega_max = 1e-5, 1e-1  # the smallest and largest tolerance of each GMRES iteration
    maxiter = 300  # maximum number of iterations
    bstep = 10  # maximum number of steps in backtracking

    # initialization
    uCurrent = J(Float64) # get the initial iteration
    omega = 0.01  # denotes the tolerance during GMRES iterations
    gamma = 0.9  # parameter of choice 2 forcing terms
    FCurrent = J(uCurrent)  # the function values of current iteration
    normFxOld = norm(FCurrent)  # the norm of residual of the last iteration
    normFxTrial = .0  # the norm of trial step, i.e., ||F(x_k) + F'(x_k)s_k||
    t = 1e-4  # restricted residual monotonicity parameter
    theta_min, theta_max = 0.1, 0.5  # the interval of backtracking steps

    # compute the final tolerance
    res_final = reltol * normFxOld + reltol  # relative tolerance and absolute tolerance

    if normFxOld < res_final
        # initial iteration is accurate enough
        return uCurrent, normFxOld
    end

    # Newton iterations begin
    for k = 1:maxiter
        if display
            @printf "No.%i iteration \n" k
            @printf "Length of current iteration: %i \n" length(uCurrent)
            @printf "Residual: %.3e \n" normFxOld
            @printf "Relative tolerance for GMRES:%.3e \n" omega
        end

        # solve for the Newton step
        delta, _, _, L = JacSolver_LU!(uCurrent, J, FCurrent, omega)

        # choose the step by backtracking
        reduced = false
        g0 = normFxOld  # g(gamma) = ||F(uCurrent + gamma * delta)||, note that the norm is not squred
        omega1 = copy(omega)
        theta1 = 1.0
        Jk_delta = L * copyto!(zeros(Float64, size(L, 1)), delta)
        gp0 = myinnerproduct(FCurrent, Jk_delta) / (-g0) # p'(0) where p is local quadratic model
        # take a trial step
        uTrial = coeffsadd_chop(true, delta, uCurrent)  # the coefficients of trial step
        # Compute the residual of trial step
        FTrial = J(uTrial)
        for i = 1:bstep
            g1 = norm(FTrial)
            if g1 < (1 - t * (1 - omega1)) * g0
                # the restricted residual monotonicity has been passed
                normFxTrial = g1
                reduced = true
                break
            else
                # choose parameter in backtracking interval using quadratic model
                # the coefficient of quadratic term
                qcoeff = g1 - g0 - gp0
                # axis of symmetry of quadratic model
                z = -gp0 / (2 * qcoeff)
                if qcoeff <= 0
                    # the direction is not a descent direction and the minimum value is at the endpoints
                    theta = theta_max
                else
                    # the minimum value is at the axis of symmetry or endpoints
                    if z > theta_max
                        theta = theta_max
                    elseif z < theta_min
                        theta = theta_min
                    else
                        theta = z
                    end
                end
            end

            # update
            rmul!(delta, theta) # backtracking
            rmul!(Jk_delta, theta) # Jacobian is linear
            gp0 *= theta # inner product is linear
            theta1 *= theta
            omega1 = 1 - theta * (1 - omega1)
            uTrial = coeffsadd_chop(true, delta, uCurrent)  # the coefficients of trial step
            FTrial = J(uTrial)
        end

        if !reduced
            # failure of backtracking and we return what we have gotten
            break
        end

        if display
            @printf "Steplength of No.%i iteration: %.3e \n \n" k theta1
        end

        if normFxTrial < res_final
            # the solution has been found
            uCurrent = uTrial
            normFxOld = normFxTrial
            break
        end

        # choose the forcing term of the next iteration
        # safeguard
        omega_safe = gamma * omega^2
        # choice 2
        omega = gamma * (normFxTrial / normFxOld)^2

        if omega_safe > 0.1
            # safeguard
            omega = max(omega, omega_safe)
        end

        # avoid the oversolving of final steps
        omega = max(omega, res_final/(2*normFxTrial))

        # ensure that omega ∈ [omega_min, omega_max]
        omega = max(omega_min, min(omega_max, omega))

        # we accept the backtracking step and update
        uCurrent = uTrial
        FCurrent = FTrial
        normFxOld = normFxTrial
        k = k + 1
    end

    if display
        @printf "Length of final solution: %i \n" length(uCurrent)
        @printf "Residual: %.3e \n" normFxOld
    end

    return uCurrent, normFxOld
end