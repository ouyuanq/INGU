function gmresr!(x0::AbstractVector{T}, res::AbstractVector{T}, Jac::Function, JacCoeffs::Vector{Vector{T}}, tol::T, kmax::Integer, maxiteration::Integer, precond::AbstractMatrix{T}, Plan, Plani) where T
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

    k, flag = gmres!(x0, res, Jac, JacCoeffs, tol, kmax, rho, precondQR, Plan, Plani)
    iteration = k
    while !flag && iteration < maxiteration
        rho = norm(res)
        k, flag = gmres!(x0, res, Jac, JacCoeffs, tol, kmax, rho, precondQR, Plan, Plani)
        iteration += k
    end
end

function gmres!(x0::AbstractVector{T}, res::AbstractVector, Jac::Function, JacCoeffs::Vector{Vector{T}}, tol::T, kmax::Integer, rho::T, precondQR, Plan, Plani) where T
    # GMRES linear equation solver
    # Input: x0 = initial iterate
    #        res = initial residual
    #        Jac, JacCoeffs = Jacobian operator
    #        tol = relative residual reduction factor
    #        kmax= max number of iterations
    #        rho = norm of initial residual
    #        precondQR = factorized preconditioner
    #        Plan, Plani = plan for fft! and ifft!
    #
    # Output: x = solution
    #         res = b - Jac(x) the residual at the end of iterations
    #         eta = the contraction factor at the end of iterations

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
    copyto!(view(v, :, 1), res)

    k = 0
    precondv = res  # store the right preconditioned result
    # GMRES iteration
    @views while(rho > tol && k < kmax)
        k += 1
        ldiv!(precondv, precondQR, v[:, k])
        Jac(v[:, k+1], JacCoeffs, precondv, Plan, Plani)
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
    ldiv!(precondQR, res)
    axpy!(true, res, x0)  # x0 .+= preconditioner\y

    # computation of the residual
    # First we compute the last column of Q where H_{m+1} = Q*R
    qmp1 = zeros(T, k+1); qmp1[end] = T(1)
    givtransapp!(qmp1, c, s, k)
    # note that res = F + J*x in this code
    @views begin
        mul!(res, v[:, 1:k+1], qmp1)
        rmul!(res, beta*qmp1[1])
    end

    converged = rho < tol ? true : false
    return k, converged
end

# in-place Givens rotations
function givapp!(c::AbstractVector{T}, s::AbstractVector{T}, vin::AbstractVector{T}) where {T}
    #  Apply a sequence of k Givens rotations, used within gmres codes
    for i in eachindex(c)
        w1 = c[i] * vin[i] - s[i] * vin[i+1]
        w2 = s[i] * vin[i] + c[i] * vin[i+1]
        vin[i] = w1
        vin[i+1] = w2
    end
end

function givtransapp!(vin::AbstractVector{T}, c::AbstractVector{T}, s::AbstractVector{T}, k::Integer) where {T}
    #  Apply a sequence of k transposed Givens rotations which created in gmres iterations to vector vin
    for i = k:-1:1
        w1 = c[i] * vin[i] + s[i] * vin[i+1]
        w2 = -s[i] * vin[i] + c[i] * vin[i+1]
        vin[i] = w1
        vin[i+1] = w2
    end
end