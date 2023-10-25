function TRC(J::Function; reltol=5e-15, inner_loops::Integer=20, display::Bool=false, mixed::Bool=true, init=zeros(0))
    # This function solves BVP operator equation F(u) = 0 approximately by
    # solving a sequence of linear operator equations J_k(du) = -F_k, where J_k = J(u_k)
    # is the Jacobian operator at iteration u_k and F_k = F(u_k). The linear
    # operator equations are discretized by ultraspherical methods and solved via
    # GMRES iterations where the tolerance can be prescribed. 
    # An almost banded matrix is used as the preconditioner.
    # The mixed precision method is added to speed up the algorithm.

    #  Input:  F = nonlinear BVP operator (multipledispatch of J)
    #          J = F' the corresponding Jacobian operator.
    #          reltol = the final relative tolerance of the residual
    #          inner_loops = maximum iterations of linear solver
    #          display = whether to show some key information during iterations
    #          mixed = whether mixed precision is used
    #
    #  Output: uCurrent = solution
    #          normFxOld = norm of residual of uCurrent 

    # key parameters
    omega_min, omega_max = 1e-5, 1e-1  # the smallest and largest tolerance of each GMRES iteration
    lambda_start = 0.1  # the initial steplength of the very first iteration
    maxiter = 800  # maximum number of iterations
    lambda_min = 1e-6  # the minimum steplength we can accept
    rho = 0.9  # quadratic convergence mode constant
    success = false  # representation of convergence or not

    # initialization
    omega = 1e-3  # the tolerance for the current GMRES iteration
     # get the initial iteration
    if isempty(init)
        uCurrent = J(Float64)
    else
        uCurrent = init
    end
    lambda = lambda_start  # steplengths of each step
    eta_k = 0.0  # the reduction factor for the current GMRES iteration
    theta_k = 0.0  # the contraction factor
    h_k = 0.0 # the estimated Kantorovich quantities
    FCurrent = J(uCurrent) # the function values of current iteration
    normFxOld = norm(FCurrent)  # the norm of residual of the last iteration
    normFxTrial = 0.0  # the norm of residual of the trial step
    chi = false  # monitor of a 'kick'
    chi2 = false
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

        # solve the linearized system
        # delta: unmodified Newton step
        # r_k: residual of GMRES iterations
        delta, _, _, r_k, eta_k, _ = JacSolver!(J, FCurrent, JacCoeffs, omega, inner_loops)

        # compute the steplength until we are satisfied
        # initial prediction
        if k > 1 && !chi2
            lambda = min(1., 1 / ((1 + eta_k) * theta_k * h_k))
        end

        acceptStep = false
        reduced = false

        while !acceptStep
            if lambda < lambda_min
                # too small a step
                if chi
                    # give up and return what we have gotten
                    return uCurrent, normFxOld
                else
                    # try a full step to jump into another convergence region
                    coeffsadd_chop!(true, delta, uCurrent)
                    FCurrent = J(uCurrent)
                    normFxOld = norm(FCurrent)
                    acceptStep = true
                    chi = true
                    chi2 = true
                    k += 1
                    lambda = lambda_start  # start from another region
                    omega = 1e-3
                    continue
                end
            else
                chi2 = false  # reset the kick

                # take a trial step
                uTrial = coeffsadd_chop(lambda, delta, uCurrent)  # the coefficients of trial step

                # Compute the residual of trial step
                FTrial = J(uTrial)  # the function values of trial step
                normFxTrial = norm(FTrial)

                # Contraction factor:
                theta_k = normFxTrial / normFxOld

                # Correction factor for the step-size:
                # t = FTrial + (lambda - 1) .* FCurrent - lambda .* r_k
                t = coeffsadd_chop(lambda - 1, FCurrent, FTrial)
                coeffsadd_chop!(-lambda, r_k, t)
                h_k = 2 * norm(t) / (lambda^2 * (1 - eta_k^2) * normFxOld)

                # If we don't observe contraction, decrease LAMBDA
                # Here we use the inexact variant of the restricted residual monotonicity test
                if theta_k >= 1 - (1 - eta_k) / 4 * lambda
                    lambda = min(1 / ((1 + eta_k) * h_k), lambda / 2)
                    reduced = true
                    # Go back to the start of the loop.
                    continue
                end

                # New potential candidate for LAMBDA
                lambdaPrime = min(1., 1 / ((1 + eta_k) * h_k))

                if lambdaPrime == 1 && lambda == 1 && normFxTrial < res_final
                    # An approximate solution has been found and a early break from the wile loop was made
                    uCurrent = uTrial
                    FCurrent = FTrial
                    normFxOld = normFxTrial
                    success = true
                    break
                end

                # Try a larger step
                if lambdaPrime >= 4 * lambda && !reduced
                    lambda = lambdaPrime
                    continue
                end

                # step accepted here
                acceptStep = true
                # We accept the trial step
                uCurrent = uTrial
                FCurrent = FTrial
                normFxOld = normFxTrial
            end
        end

        if display
            @printf "Steplength of No.%i iteration: %.3e \n \n" k lambda
        end

        if success || normFxOld < res_final
            # the solution has been found
            break
        end

        if !chi2
            # prepare for the next step
            h_quad = 2 * rho * theta_k^2 / ((1 + rho) * (1 - eta_k^2))
            omega_quad = min((sqrt(1 + h_quad^2) - 1) / h_quad, omega_max)
            # Quadratic convergence mode
            omega = max(omega_quad, omega_min)  # assure that ω is not that small
        end

        # avoid the oversolving of final steps
        omega = max(omega, res_final/(8 * normFxTrial))

        # update the counter
        k = k + 1
    end

    if display
        @printf "Length of final solution: %i \n" length(uCurrent)
        @printf "Residual: %.3e \n" normFxOld
    end

    return uCurrent, normFxOld
end

function TRC_LU(J::Function; reltol=5e-15, display::Bool=false, init=zeros(0))
    # Same as TRC but replace the inner iterations (GMRES) by direct method (linsolve!)

    # key parameters
    omega_min, omega_max = 1e-5, 1e-1  # the smallest and largest tolerance of each GMRES iteration
    lambda_start = 0.1  # the initial steplength of the very first iteration
    maxiter = 800  # maximum number of iterations
    lambda_min = 1e-6  # the minimum steplength we can accept
    rho = 0.9  # quadratic convergence mode constant
    success = false  # representation of convergence or not

    # initialization
    omega = 1e-3  # the tolerance for the current GMRES iteration
     # get the initial iteration
    if isempty(init)
        uCurrent = J(Float64)
    else
        uCurrent = init
    end
    lambda = lambda_start  # steplengths of each step
    eta_k = 0.0  # the reduction factor for the current GMRES iteration
    theta_k = 0.0  # the contraction factor
    h_k = 0.0 # the estimated Kantorovich quantities
    FCurrent = J(uCurrent) # the function values of current iteration
    normFxOld = norm(FCurrent)  # the norm of residual of the last iteration
    normFxTrial = 0.0  # the norm of residual of the trial step
    chi = false  # monitor of a 'kick'
    chi2 = false
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

        # solve the linearized system
        delta, r_k, eta_k, _ = JacSolver_LU!(uCurrent, J, FCurrent, omega)

        # compute the steplength until we are satisfied
        # initial prediction
        if k > 1 && !chi2
            lambda = min(1., 1 / ((1 + eta_k) * theta_k * h_k))
        end

        acceptStep = false
        reduced = false

        while !acceptStep
            if lambda < lambda_min
                # too small a step
                if chi
                    # give up and return what we have gotten
                    return uCurrent, normFxOld
                else
                    # try a full step to jump into another convergence region
                    coeffsadd_chop!(true, delta, uCurrent)
                    FCurrent = J(uCurrent)
                    normFxOld = norm(FCurrent)
                    acceptStep = true
                    chi = true
                    chi2 = true
                    k += 1
                    lambda = lambda_start  # start from another region
                    omega = 1e-3
                    continue
                end
            else
                chi2 = false  # reset the kick

                # take a trial step
                uTrial = coeffsadd_chop(lambda, delta, uCurrent)  # the coefficients of trial step

                # Compute the residual of trial step
                FTrial = J(uTrial)  # the function values of trial step
                normFxTrial = norm(FTrial)

                # Contraction factor:
                theta_k = normFxTrial / normFxOld

                # Correction factor for the step-size:
                # t = FTrial + (lambda - 1) .* FCurrent - lambda .* r_k
                t = coeffsadd_chop(lambda - 1, FCurrent, FTrial)
                coeffsadd_chop!(-lambda, r_k, t)
                h_k = 2 * norm(t) / (lambda^2 * (1 - eta_k^2) * normFxOld)

                # If we don't observe contraction, decrease LAMBDA
                # Here we use the inexact variant of the restricted residual monotonicity test
                if theta_k >= 1 - (1 - eta_k) / 4 * lambda
                    lambda = min(1 / ((1 + eta_k) * h_k), lambda / 2)
                    reduced = true
                    # Go back to the start of the loop.
                    continue
                end

                # New potential candidate for LAMBDA
                lambdaPrime = min(1., 1 / ((1 + eta_k) * h_k))

                if lambdaPrime == 1 && lambda == 1 && normFxTrial < res_final
                    # An approximate solution has been found and a early break from the wile loop was made
                    uCurrent = uTrial
                    FCurrent = FTrial
                    normFxOld = normFxTrial
                    success = true
                    break
                end

                # Try a larger step
                if lambdaPrime >= 4 * lambda && !reduced
                    lambda = lambdaPrime
                    continue
                end

                # step accepted here
                acceptStep = true
                # We accept the trial step
                uCurrent = uTrial
                FCurrent = FTrial
                normFxOld = normFxTrial
            end
        end

        if display
            @printf "Steplength of No.%i iteration: %.3e \n \n" k lambda
        end

        if success || normFxOld < res_final
            # the solution has been found
            break
        end

        if !chi2
            # prepare for the next step
            h_quad = 2 * rho * theta_k^2 / ((1 + rho) * (1 - eta_k^2))
            omega_quad = min((sqrt(1 + h_quad^2) - 1) / h_quad, omega_max)
            # Quadratic convergence mode
            omega = max(omega_quad, omega_min)  # assure that ω is not that small
        end

        # avoid the oversolving of final steps
        omega = max(omega, res_final/(8 * normFxTrial))

        # update the counter
        k = k + 1
    end

    if display
        @printf "Length of final solution: %i \n" length(uCurrent)
        @printf "Residual: %.3e \n" normFxOld
    end

    return uCurrent, normFxOld
end