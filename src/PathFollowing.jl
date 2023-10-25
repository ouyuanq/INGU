# solve the singularly perturbed problems by path-following methods
function pathfollowing(N::Function, Dlam::Function, lambda0::AbstractFloat, lambda1::AbstractFloat)
    # Inputs: N = nonlinear BVP operator and its Jacobian
    #         Dlam = derivative with respect to lambda
    #         lambda0 = initial parameter with which the nonlinear problem is easy to solve
    #         lambda1 = final parameter
    # Note: lambda0 is always big enough to star with and we desire the solution when lambda is small. Therefore, we always take the negative tangent in the first path-following, i.e., we want decrease in lambda.

    # key parameters
    stepmin = 1e-6 # the minimum steplength allowed

    # initialization
    T = Float64
    retract = false # whether we shrink the steplength or not
    kmax = 100 # the maximum number of solutions(in case of dead loops)
    sl = (lambda0 - lambda1) / 2 # initial step for tangent vector

    # solve the initial nonlinear BVP
    uold, _ = TRC(setepsilon(N, lambda0); reltol = 1e-3)
    lamold = Float32(lambda0)
    upre = zeros(T, 0)
    utangent = zeros(Float32, 0) # initial tangent for u
    lamtangent = -one(Float32) # initial tangent for lambda(-1 if we want decrease and +1 otherwise)

    @printf "lambda: %.4e \n" lamold
    @printf "length: %i \n" length(uold)

    # store the solutions and parameters along the path
    uvec = [copy(uold)]
    lamvec = [lamold]
    utangentvec = [copy(utangent)]
    lamtangentvec = [lamtangent]
    uprevec = Vector{typeof(uold)}(undef, 0)
    lamprevec = Vector{typeof(lamold)}(undef, 0)

    # Start the pseudo-arclength path following algorithm. It proceeds as follows:
    #   1. Find a tangent direction.
    #   2. Move in the direction of the tangent for the given steplength.
    #   3. Compute the Newton correction.
    #   4. If Newton was happy, accept the new point on the curve. If not, shrink
    #      the steplength by a factor of 4 and go back to step 2.
    for k = 1:kmax
        @printf "No.%i iteration \n" k
        @printf "sl: %.4e \n" sl
        # Find a tangent direction, but only if we were not to retract an old direction. At the start of the while loop, recall that RETRACT == false.
        if !retract
            # solve the Jacobian system extended by pseudo-arlength condition to get a tangent
            lamtangent, utangent = JacSolver_tangent!(N, Dlam, uold, lamold, utangent, lamtangent)
            # normlize the tangent
            scale = sqrt(sum(abs2, utangent) + lamtangent ^ 2)
            lamtangent /= scale
            ldiv!(scale, utangent)
            push!(lamtangentvec, lamtangent)
            push!(utangentvec, copy(utangent))
        end
        # move in the direction of the tangent, i.e., compute a predictor
        upre = coeffsadd_chop(sl, utangent, uold)
        lampre = lamold + Float32(sl) * lamtangent
        push!(uprevec, copy(upre))
        push!(lamprevec, lampre)

        # Do Newton correction to get back on the solution curve
        @printf "lambda & length before Newton correction:%.4e \n" lampre
        lampre, NewtonIter, retract = Newton_corrector!(N, Dlam, upre, lampre, utangent, lamtangent)
        @printf "lambda & after Newton correction:%.4e \n" lampre

        # If the Newton correction algorithm told us we were trying to take too long tangent steps, we decrease the steplength.
        if retract
            # @printf "retract \n"
            error("retract")
            sl *= 0.25

            if sl < stepmin
                @printf "Failed: steplength too small"
                break
            end

            continue
        end

        # We've found a new point, update iteration:
        uold = upre
        lamold = lampre
        @printf "lambda: %.4e \n" lamold
        @printf "length: %i \n" length(uold)
        push!(uvec, copy(uold))
        push!(lamvec, lamold)

        if lamold < lambda1 * (1 + 2e-2)
            @printf "Happy result. \n"
            # we get a high resolution result
            # polish the last iteration to get the final solution
            ufinal, _ = TRC(setepsilon(N, lambda1); reltol = 1e-14, init = uold, display = true)
            push!(uvec, copy(ufinal))
            push!(lamvec, lambda1)
            return uvec, lamvec, utangentvec, lamtangentvec, uprevec, lamprevec
        end

        # If we're experiencing good Newton convergence, we try to get the steplength closer to the maximum steplength allowed:
        if NewtonIter < 3
            sl = min(sl, (lamold - lambda1) / 1.2)
        end
    end
    uvec, lamvec, utangentvec, lamtangentvec, uprevec, lamprevec
end

function setepsilon(N::Function, l::AbstractFloat)
    # bind the parameter l to the nonlinear operator n
    G(::Type{T}) where T = N(T)
    G(u::AbstractVector{T}) where {T} = N(u; epsilon=l)
    G(u::AbstractVector{T}, ::AbstractChar;) where {T} = N(u, 'C'; epsilon=l)
    G(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, VecPlan, VecPlani) where {T} = N(g, JacCoeffs, delta, VecPlan, VecPlani)
    G(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T = N('T', JacCoeffs, FC, Plan, Plani)
    G(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where {T} = N('B', bc, n)

    G
end

function JacSolver_tangent!(N::Function, Dlam::Function, uold::AbstractVector, lamold::AbstractFloat, utangent::AbstractVector, lamtangent::AbstractFloat; reltol=5e-3, b::AbstractVector=zeros(0))
    # compute the approximate tangent of N at (uold, lamold), i.e., solve an extended Jacobian system
    # [      lamtangent           <utangent, ⋅ >    ] [tau] = 1 - b[1]
    # [ dN/dlam(uold, lamold)  dN/du(uold, lamold) ] [ t ] = [0] - b[2:end]
    T = Float32  # mixed precision used
    G = setepsilon(N, lamold)

    n = max(length(utangent), length(uold), length(b), 16)
    n = ceil(Integer, 1.25 * n) + 5

    JacCoeffs = G(convert(Vector{T}, uold), 'C')
    t = zeros(T, n+1)  # dimension n+1
    res = zeros(T, n+1)
    res[1] = one(T)  # pseudo-arclength condition at the first equation
    coeffsadd!(-one(T), b, res)

    # get the preconditioner (note that p may be decreased for first Newton iteration)
    p = floor(Integer, sqrt(log2(n))) # the fixed bandwidth of each term in preconditioner
    # decrease p if all coefficients are short(i.e., p could be less than sqrt(log2(n)))
    p = min(p, max(0, maximum(length.(JacCoeffs) .- (1:length(JacCoeffs)))))
    PreCoeffs = trunc(JacCoeffs, p)
    N = length(JacCoeffs) - 1  # the differential order
    bc = zeros(eltype(res), 0, N)  # boundary conditions
    bc = G('B', bc, n)
    precond_band = bandinit(PreCoeffs, n, p)  # operator parts

    # plan for fft
    n_fft = 2 * max(n + N - 2, n)
    Plan = plan_fft!(Vector{Complex{T}}(undef, n_fft))
    Plani = plan_ifft!(Vector{Complex{T}}(undef, n_fft))
    Dlamuold = Dlam(convert(Vector{T}, uold))
    if length(Dlamuold) < n
        append!(Dlamuold, zeros(T, n - length(Dlamuold)))
    end
    while true
        # preconditioner
        precond = almostbanded_init(precond_band, bc, p)

        # solve!
        # Note that JacCoeffs, Dlamuold, utangent and lamtangent contain all information about extended Jacobian
        gmresr_tangent!(t, res, G, JacCoeffs, Dlamuold, utangent, lamtangent, reltol * norm(res), min(max(20, fld(n, 100)), 150), 150, precond, Plan, Plani)

        @views cutoff = standardChop(t[2:n+1], 5 * eps(T))
        if cutoff < n
            # we get a satisfying solution
            deleteat!(t, cutoff+2:n+1)
            break
        else
            # the solution is not resolved enough so we double its length
            append!(t, zeros(T, n))  # warm restart
            append!(res, Vector{T}(undef, n - 1))
            n *= 2

            # redo the plans for FFT
            n_fft = 2 * max(n + N - 2, n)
            Plan = plan_fft!(Vector{Complex{T}}(undef, n_fft))
            Plani = plan_ifft!(Vector{Complex{T}}(undef, n_fft))

            # prolong dN/(dlambda) if necessary
            if length(Dlamuold) < n
                append!(Dlamuold, zeros(T, n - length(Dlamuold)))
            end

            # recompute the reisudal: [1; \bold{0}] - extendedJac[JacCoeffs, Dlamuold, utangent, lamtangent](t)
            G(res, JacCoeffs, view(t, 2:n+1), Plan, Plani)
            @views axpy!(t[1], Dlamuold[1:n], res)
            rmul!(res, -one(eltype(res)))
            @views prepend!(res, 1-myinnerproduct(utangent, t[2:n+1]) - lamtangent * t[1])  # psedo-arclength cond
            coeffsadd!(-one(T), b, res)

            # expand the preconditioner
            precond_band = bandexpand(precond_band, PreCoeffs, n, p)
            bc = G('B', bc, n)
        end
    end
    # tangents for lambda and u, respectively
    tau = popfirst!(t)

    tau, t
end

function gmresr_tangent!(x0::AbstractVector{T}, res::AbstractVector{T}, G::Function, JacCoeffs::Vector{Vector{T}}, Dlamuold::AbstractVector{T}, utangent::AbstractVector, lamtangent::AbstractFloat, tol::AbstractFloat, kmax::Integer, maxiteration::Integer, precond::AlmostBandedMatrix{T}, Plan, Plani) where T
    # restart GMRES with restart number kmax

    # test for initial iteration
    rho = norm(res)
    if rho < tol
        # the initial iteration is an approximate solution
        return
    end

    flag = false
    # preconditioner factorization (precond is not recycled)
    precondQR = qr!(precond)

    k, flag = gmres_tangent!(x0, res, G, JacCoeffs, Dlamuold, utangent, lamtangent, tol, kmax, rho, precondQR, Plan, Plani)
    iteration = k
    while !flag && iteration < maxiteration
        rho = norm(res)
        k, flag = gmres_tangent!(x0, res, G, JacCoeffs, Dlamuold, utangent, lamtangent, tol, kmax, rho, precondQR, Plan, Plani)
        iteration += k
    end
end

function gmres_tangent!(x0::AbstractVector{T}, res::AbstractVector{T}, G::Function, JacCoeffs::Vector{Vector{T}}, Dlamuold::AbstractVector{T}, utangent::AbstractVector, lamtangent::AbstractFloat, tol::AbstractFloat, kmax::Integer, rho::T, precondQR, Plan, Plani) where T
    # initialization
    n = length(res)

    h = Matrix{T}(undef, kmax + 1, kmax)
    v = Matrix{T}(undef, n, kmax + 1)
    c = Vector{T}(undef, kmax+1)
    s = Vector{T}(undef, kmax+1)

    # residual is already computed and stored in vector res
    beta = copy(rho)  # beta = norm(r_0)
    g = zeros(T, kmax + 1)
    g[1] = rho

    rdiv!(res, rho)
    v[:, 1] .= res

    k = 0
    precondv = view(res, 2:n)  # store the right preconditioned result
    # GMRES iteration
    @views while(rho > tol && k < kmax)
        k += 1
        ldiv!(precondv, precondQR, v[2:n, k])
        G(v[2:n, k+1], JacCoeffs, precondv, Plan, Plani) # dN/du
        axpy!(v[1, k], Dlamuold[1:n-1], v[2:n, k+1])  # dN/dlam
        v[1, k+1] = myinnerproduct(precondv, utangent) + T(lamtangent) * v[1, k]
        normav = norm(v[:, k+1])

        # Modified Gram-Schmidt
        @views @inbounds for j = 1:k
            h[j, k] = dot(v[:, j], v[:, k+1])
            axpy!(-h[j, k], v[:, j], v[:, k+1])
        end
        h[k+1, k] = norm(v[:, k+1])

        # Reorthogonalize if necessary
        if normav + .001*h[k+1, k] == normav
            @views @inbounds for j = 1:k
                hr = dot(v[:, j], v[:, k+1])
                h[j, k] += hr
                axpy!(-hr, v[:, j], v[:, k+1])
            end
            h[k+1, k] = norm(view(v, :, k+1))
        end

        # TODO: watch out for happy breakdown
        @views if h[k+1, k] != 0
            rdiv!(v[:, k+1], h[k+1, k])
        end

        # Form and store the information for the new Givens rotation
        if k > 1
            @views givapp!(c[1:k-1], s[1:k-1], h[1:k, k])
        end
        nu = norm(h[k:k+1, k])
        if nu != 0
            c[k] = h[k, k]/nu
            s[k] = -h[k+1, k]/nu
            h[k, k] = c[k]*h[k, k] - s[k]*h[k+1, k]
            h[k+1, k] = 0
            @views givapp!(c[k:k], s[k:k], g[k:k+1])
        end

        # Update the norm of residual
        rho = abs(g[k+1])
    end

    # Solve for final result since either the iteration converges or the maximum number of iterations is reached
    LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', view(h, 1:k, 1:k), view(g, 1:k))
    mul!(res, view(v, :, 1:k), view(g, 1:k))  # res is used as a temporary vector
    ldiv!(precondQR, view(res, 2:n))
    axpy!(true, res, x0)

    # computation of the residual
    # First we compute the last column of Q where H_{m+1} = Q*R
    qmp1 = zeros(T, k+1); qmp1[end] = T(1)
    givtransapp!(qmp1, c, s, k)
    @views begin
        mul!(res, v[:, 1:k+1], qmp1)
        rmul!(res, beta*qmp1[1])
    end

    converged = rho < tol ? true : false
    return k, converged
end

function Newton_corrector!(N::Function, Dlam::Function, upre::AbstractVector, lampre::AbstractFloat, utangent::AbstractVector, lamtangent::AbstractFloat; reltol::Float64 = 1e-2)
    # apply Newton iterations to solve nonlinear BVP with initial iteration [uold; lamold]
    retract = false
    b = N(upre; epsilon = lampre) # rhs
    prepend!(b, one(eltype(b)))

    # do 6 Newton iterations
    for k = 1:6
        nrmu = norm(upre)

        # solve the extended Jacobian system to get a Newton correction
        dlam, du = JacSolver_tangent!(N, Dlam, upre, lampre, utangent, lamtangent; reltol=5e-3, b=b)

        # take a full Newton step since good initial iteration is assumed
        lampre += dlam
        coeffsadd_chop!(true, du, upre)

        # test convergence of Newton iteration
        nrmdu = sqrt(sum(abs2, du) + dlam^2)
        if nrmdu / nrmu < reltol
            # converged
            return lampre, k, retract
        end
        # We wanted to take too many iterations, so tell the followpath algorithm to retract and take a smaller tangent step.
        if k == 6
            retract = true
            return lampre, k, retract
        end
    end
end