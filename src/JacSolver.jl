# solve the linearized equation, i.e. the Jacobian, of nonlinear equation at some uk

function JacSolver!(Jac::Function, FCurrent::AbstractVector, JacCoeffs::Vector, tol::AbstractFloat, inner_loops::Integer)
    # solve the linearized Jacobian system Jac[uCurrent](x) = FCurrent approximately by GMRES
    # Jac is a function handle which represents the linear operator and an almost-banded matrix is used as preconditioner
    # Note that the tolerance of GMRES is tol and JacCoeffs is already computed

    # Mixed precision used
    T = eltype(eltype(JacCoeffs))
    lF = length(FCurrent)
    N = length(JacCoeffs) - 1  # the differential order

    # the size of linear equation
    n = max(length(JacCoeffs) + maximum(length.(JacCoeffs) .- (1:length(JacCoeffs))), standardChop(FCurrent, eps(T)))
    n = max(ceil(Integer, 1.25 * n) + 5, 16)  # extra length reserved for plateau
    n += mod(n, 2)  # make n an even number
    n += min(0, 2-N)  # compensation for exact truncation
    p = floor(Integer, sqrt(log2(n))) # the fixed bandwidth of each term in preconditioner
    # decrease p if all coefficients are short(i.e., p could be less than sqrt(log2(n)))
    p = min(p, max(0, maximum(length.(JacCoeffs) .- (1:length(JacCoeffs)))))

    # We suspect that the solution to the Jacobian system is something like
    # zeros so that we have reached the stationary point. 
    # If the Newton step is not resolved, the solution of last GMRES iterations 
    # can be used as the initial iteration for the next GMRES iteration.
    xinitial = zeros(T, n)
    res = copyto!(zeros(T, n), view(FCurrent, 1:min(n, lF))) # zeros initial iteration assumed
    tol_res = T(tol) * norm(res)  # relative tolerance

    # get the preconditioner (note that p may be decreased for first Newton iteration)
    PreCoeffs = trunc(JacCoeffs, p)
    bc = zeros(T, 0, N)  # boundary conditions
    bc = Jac('B', bc, n)
    precond_band = bandinit(PreCoeffs, n, p)  # operator parts

    # plan for fft
    n_fft = 2 * max(n + N - 2, n)
    Plan = plan_fft!(Vector{Complex{T}}(undef, n_fft))
    Plani = plan_ifft!(Vector{Complex{T}}(undef, n_fft))
    inter_iter = 1
    while true
        # preconditioner
        precond = almostbanded_init(precond_band, bc, p)

        # solve!
        gmresr!(xinitial, res, Jac, JacCoeffs, tol_res, min(max(inner_loops, fld(n, 100)), 150), 150, precond, Plan, Plani)

        # test convergence
        cutoff = standardChop(xinitial, 5 * eps(T))
        if cutoff < n
            # we get a satisfying solution
            deleteat!(xinitial, cutoff+1:n)
            rmul!(xinitial, -one(T))
            break
        else
            # the solution is not resolved enough so we double its length
            append!(xinitial, zeros(T, n))   # warm restart
            append!(res, Vector{T}(undef, n))
            n *= 2

            # redo the plans for FFT
            n_fft = 2 * max(n + N - 2, n)
            Plan = plan_fft!(Vector{Complex{T}}(undef, n_fft))
            Plani = plan_ifft!(Vector{Complex{T}}(undef, n_fft))

            # recompute the reisudal: b - Jac[JacCoeffs](xinitial)
            Jac(res, JacCoeffs, xinitial, Plan, Plani)
            rmul!(res, -one(T))
            coeffsadd!(true, view(FCurrent, 1:min(n, lF)), res)

            # expand the preconditioner
            precond_band = bandexpand(precond_band, PreCoeffs, n, p)
            bc = Jac('B', bc, n)
            inter_iter += 1
        end
    end

    return xinitial, Plan, Plani, res, norm(res)/norm(FCurrent), inter_iter
end


function JacSolver_LU!(uCurrent::AbstractVector{T}, Jac::Function, b::AbstractVector{T}, tol::AbstractFloat) where T
    # solve a linear differential equation Jac[uCurrent](delta) = b directly by LU factorization, where b is given
    # Jac is a function handle which represents the linear operator

    # get the coefficients
    JacCoeffs = Jac(copy(uCurrent), 'C')
    N = length(JacCoeffs) - 1  # maximum differential order

    # the size of linear equation
    lb = length(b)
    n = max(length(JacCoeffs) + maximum(length.(JacCoeffs) .- (1:length(JacCoeffs))), lb)
    n = max(ceil(Integer, 1.25 * n) + 5, 16)   # extra length reserved for plateau

    # initialization
    eta = zero(T)
    delta = zeros(T, 0)
    res = Vector{T}(undef, n)
    bc = Jac('B', zeros(T, 0, N), n)  # boundary conditions
    L = zeros(T, 0, 0)
    # Loop over a finer and finer grid until happy
    while true
        # get the matirx of differential equation
        L = matrix(JacCoeffs, bc)

        # get the rhs
        copyto!(res, view(b, 1:min(lb, n)))
        res[lb+1:n] .= 0

        # Solve the linear system by standard LinearAlgebra provided by Julia
        delta = L \ res

        # residual
        mul!(res, L, delta, -1, true)

        # test convergence
        cutoff = standardChop(delta, 45 * eps(T))
        eta = norm(res) / norm(b)
        if cutoff < n && eta < tol
            # we get a satisfying solution
            deleteat!(delta, cutoff+1:n)
            rmul!(delta, -1)  # negative output since b = F(uCurrent)
            break
        else
            # the solution is not resolved enough so we double its length
            append!(res, Vector{T}(undef, n))
            append!(delta, zeros(T, n))
            n *= 2
            bc = Jac('B', bc, n)
        end
    end

    delta, res, norm(res) / norm(b), L
end