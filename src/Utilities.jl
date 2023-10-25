# the auxiliary functions used in Newton method
using FFTW, ToeplitzMatrices

# addition of two coefficients
function coeffsadd(alpha::Number, x::AbstractVector, y::AbstractVector)
    # return the coefficient alpha * x + y
    lenx, leny = length(x), length(y)
    out = append!(copy(y), zeros(eltype(y), max(lenx, leny) - leny))
    axpy!(alpha, x, view(out, 1:lenx))

    out
end

function coeffsadd!(alpha::Number, x::AbstractVector, y::AbstractVector)
    # compute the coefficient alpha * x + y and store the result in y
    lenx, leny = length(x), length(y)
    append!(y, zeros(eltype(y), max(lenx, leny) - leny))
    axpy!(alpha, x, view(y, 1:lenx))

    y
end

function coeffsadd_chop(alpha::Number, x::AbstractVector, y::AbstractVector)
    # return the coefficient alpha * x + y
    lenx, leny = length(x), length(y)
    out = append!(copy(y), zeros(eltype(y), max(lenx, leny) - leny))
    axpy!(alpha, x, view(out, 1:lenx))

    deleteat!(out, standardChop(out)+1:length(out))
end

function coeffsadd_chop!(alpha::Number, x::AbstractVector, y::AbstractVector)
    # compute the coefficient alpha * x + y and store the result in y
    lenx, leny = length(x), length(y)
    append!(y, zeros(eltype(y), max(lenx, leny) - leny))
    axpy!(alpha, x, view(y, 1:lenx))

    deleteat!(y, standardChop(y)+1:length(y))
end

# inner product of two vectors of different lengths
function myinnerproduct(a::AbstractVector{T}, b::AbstractVector) where {T}
    # compute the inner product of a and b where we append the shorter one with
    # zeros to accommodate the longer one
    # for loop in JULIA language!
    uv = zero(T)
    if length(a) < length(b)
        @inbounds for i in eachindex(a)
            uv += a[i] * b[i]
        end
    else
        @inbounds for i in eachindex(b)
            uv += a[i] * b[i]
        end
    end

    uv
end

# in-place evaluation of Jacobian-vector product
function JacEval!(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where {T}
    # g = J[JacCoeffs] * delta where the linear operator is JacCoeffs[N] * delta^(N-1) + ... + JacCoeffs[2] * delta' + JacCoeffs[1] * delta and N = length(JacCoeffs)
    # Note that JacCoeffs are all Chebyshev series and plan is given for fast FFTs
    # For economy, all terms are added up before inverse FFT is applied, where one third computational cost can be saved.

    N = length(JacCoeffs) - 1
    n_fft = length(Plan)
    complex_delta = Vector{Complex{T}}(undef, n_fft)  # vectors to be applied by FFTs
    complex_coeff = Vector{Complex{T}}(undef, n_fft)
    complex_g = zeros(Complex{T}, n_fft)
    dtemp = fastconv!(copy(delta), 0, 1)
    gtemp = zeros(T, cld(n_fft, 2))
    # the 0th order term
    if isempty(JacCoeffs[1])
        # do nothing
    elseif length(JacCoeffs[1]) < 3
        # no need for FFTs
        coeffsadd!(true, fastmult1!(dtemp, JacCoeffs[1]), gtemp)
    else
        fastmult1_eco!(complex_g, complex_delta, complex_coeff, dtemp, JacCoeffs[1], Plan)
    end

    constantterm = zeros(Integer, 0)
    for i = 1:N
        if length(JacCoeffs[i+1]) == 1
            if !iszero(JacCoeffs[i+1])
                push!(constantterm, i)  # nonzero constant term
            end
        elseif length(JacCoeffs[i+1]) == 2
            # no plan is needed
            fastdive2U!(dtemp, delta, i)  # inversion and differentiation
            coeffsadd!(true, fastmult1!(dtemp, JacCoeffs[i+1]), gtemp)
        elseif !isempty(JacCoeffs[i+1])
            fastdive2U!(dtemp, delta, i)  # inversion and differentiation
            fastmult1_eco!(complex_g, complex_delta, complex_coeff, dtemp, JacCoeffs[i+1], Plan)
        end
    end
    if !iszero(complex_g)
        fastmult1_eco!(gtemp, complex_g, Plani) # apply inverse FFT
    end

    gtemp_index = 1  # gtemp is a C^{(1)} basis
    for i in constantterm
        fastconv!(gtemp, gtemp_index, i)
        gtemp_index = i  # gtemp is a C^{(i)} basis now
        # nonzero constant term
        coeffsadd!(JacCoeffs[i+1][1], fastdiff!(dtemp, delta, i), gtemp)
    end
    fastconv!(gtemp, gtemp_index, N)  # convert to C^{(N)} basis for output

    # note that n_fft > n by default
    copyto!(g, view(gtemp, 1:length(g)))
end

# product of transpose of Jacobian and vectors
function JacEval_trans(JacCoeffs::Vector{Vector{T}}, delta::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where {T}
    # g = (J[JacCoeffs])^T * delta where the linear operator is JacCoeffs[N] * delta^(N-1) + ... + JacCoeffs[2] * delta' + JacCoeffs[1] * delta and N = length(JacCoeffs)
    # Note that JacCoeffs are all Chebyshev series and plans are given for fast FFTs

    N, lendelta = length(JacCoeffs) - 1, length(delta)
    n_fft = length(Plan)
    n_g = cld(n_fft, 2)
    # common factor for variable coefficients
    n1 = fastconv_trans(delta, 1, N, T)  #transpose of conversion S_{N-1}...S_1
    if length(n1) > n_g
        # fit for FFT plans
        deleteat!(n1, n_g+1:length(n1))
    end

    # common factors for fast matrix-vector multiplication via FFTs
    complex_deltaT = Vector{Complex{T}}(undef, n_fft)  # FFT vector for Toeplitz matrices
    copyto!(complex_deltaT, n1)
    complex_deltaT[lendelta+1:end] .= 0
    complex_deltaH = Vector{Complex{T}}(undef, n_fft)  # FFT vector for Hankel matrices
    complex_deltaH[1:n_g-lendelta] .= 0
    copyto!(view(complex_deltaH, n_g-lendelta+1:n_g), view(n1, lendelta:-1:1))
    complex_deltaH[n_g+1:end] .= 0
    Plan * complex_deltaT
    Plan * complex_deltaH

    complex_coeff = Vector{Complex{T}}(undef, n_fft)
    complex_g = zeros(Complex{T}, n_fft)

    # the 0th order term (note that S_0 M_0 = M_1 S_0)
    if isempty(JacCoeffs[1])
        g = zeros(T, 0)
    elseif length(JacCoeffs[1]) < 3
        # no need for FFTs
        g = fastmult1!(copy(n1), JacCoeffs[1])
    else
        fastmult1_eco!(complex_g, complex_deltaT, complex_deltaH, complex_coeff, JacCoeffs[1], Plan)
        g = fastmult1_eco!(zeros(T, n_g), complex_g, Plani) # apply inverse FFT
    end
    fastconv_trans!(g, 0)  # apply the transpose of S_0

    constantterm = zeros(Integer, 0)
    JacTdelta = Vector{T}(undef, n_g)
    for i = 1:N
        if length(JacCoeffs[i+1]) == 1
            if !iszero(JacCoeffs[i+1][1])
                push!(constantterm, i)
            end
        elseif length(JacCoeffs[i+1]) == 2
            # no plan is needed
            coeffsadd!(true, fastinff4U!(fastmult1!(copy(n1), JacCoeffs[i+1]), i), g)
        elseif !isempty(JacCoeffs[i+1])
            complex_g .= 0
            JacTdelta .= 0
            fastmult1_eco!(complex_g, complex_deltaT, complex_deltaH, complex_coeff, JacCoeffs[i+1], Plan)
            fastmult1_eco!(JacTdelta, complex_g, Plani) # apply inverse FFT
            coeffsadd!(true, fastinff4U!(JacTdelta, i), g)
        end
    end

    for i in constantterm
        # nonzero constant term
        coeffsadd!(JacCoeffs[i+1][1], fastdiff_trans!(fastconv_trans(delta, i, N, T), i), g)
    end

    # trim g if necessary
    deleteat!(g, n_g+1:length(g))
end

## construction for coefficients of preconditioner
function lowdegapprox(JacCoeffs::Vector{Vector{T}}, p::Integer) where T
    # compute the best low degree approximation of JacCoeffs
    # Note: there are many criterions for choosing low degree polynomials and we take the better one between truncation and aliasing which is more close to the iniital polynomials in L^2 sense. 
    PreCoeffs = similar(JacCoeffs)
    lenJac = length.(JacCoeffs)

    # get the matrix for judging L^2 norm
    A = minL2coeffs(maximum(lenJac), p+length(JacCoeffs), T)
    for i in eachindex(JacCoeffs)
        bw = p+i  # bandwidth
        if lenJac[i] > bw
            # initial polynomial is of high degree and we choose the better one
            trunci = JacCoeffs[i][1:bw]
            aliasi = alias(JacCoeffs[i], bw-1)

            rhs = view(A, 1:bw, bw+1:lenJac[i]) * view(JacCoeffs[i], bw+1:lenJac[i])
            trunc_res = norm(rhs)  # x = 0 for truncation
            tadiff = trunci - aliasi
            mul!(rhs, view(A, 1:bw, 1:bw), tadiff, true, true)
            alias_res = norm(rhs)
            if trunc_res < alias_res
                # truncation is better
                PreCoeffs[i] = trunci
            else
                # aliasing is better
                PreCoeffs[i] = aliasi
            end
        else
            # initial polynomial is of low degree and we take it as coefficient of preconditioner
            PreCoeffs[i] = copy(JacCoeffs[i])
        end

        # preparation for preconditioner construction (division by 2 except the first element)
        PreCoeffs[i][1] *= 2
        ldiv!(T(2), PreCoeffs[i])
    end

    PreCoeffs, p
end

function trunc(JacCoeffs::Vector{Vector{T}}, p::Integer) where {T}
    # truncate the coefficients of Jacobian system for precodnitioner
    # NOTE: p may be less than sqrt(log2(n)) since the lengths of coefficients in the first Newton iteration are often short.
    # NOTE: coefficients are divided by 2 except for the first element which is convenient for construction of preconditioner
    PreCoeffs = similar(JacCoeffs)

    # truncation
    for i in eachindex(JacCoeffs)
        if length(JacCoeffs[i]) > p+i
            PreCoeffs[i] = JacCoeffs[i][1:p+i]
        else
            PreCoeffs[i] = copy(JacCoeffs[i])
        end

        # divide for preconditioner construction
        if !isempty(PreCoeffs[i])
            PreCoeffs[i][1] *= 2
            ldiv!(T(2), PreCoeffs[i])
        end
    end

    PreCoeffs
end

function alias(JacCoeffs::Vector{Vector{T}}, p::Integer) where {T}
    # alias the coefficients for preconditioner
    # NOTE: p may be less than sqrt(log2(n)) since the lengths of coefficients in the first Newton iteration are often short.
    # NOTE: coefficients are divided by 2 except for the first element which is convenient for construction of preconditioner
    PreCoeffs = similar(JacCoeffs)
    lenJac = length.(JacCoeffs)

    # aliasing
    for i in eachindex(JacCoeffs)
        PreCoeffs[i] = alias(JacCoeffs[i], p + i - 1)

        # if JacCoeffs[i] is too short, there is nothing lost in preconditioner
        if lenJac[i] > p + i
            # determine whether truncation is better (important for some cases!)
            truncdiff = copy(JacCoeffs[i])
            truncdiff[1:p+i] .= zero(T)
            aliasdiff = coeffsadd(-one(T), PreCoeffs[i], JacCoeffs[i])
            # FFT plan preparation
            Plan = plan_fft!(Vector{Complex{T}}(undef, 4 * lenJac[i] - 3))
            Plani = plan_ifft!(Vector{Complex{T}}(undef, 4 * lenJac[i] - 3))
            # L2 norm
            fastsquare!(truncdiff, Plan, Plani)
            fastsquare!(aliasdiff, Plan, Plani)

            if fastquad(truncdiff) < fastquad(aliasdiff)
                # trunction is better
                PreCoeffs[i] = JacCoeffs[i][1:p+i]
            end
        end

        # divide for preconditioner construction
        if !isempty(PreCoeffs[i])
            PreCoeffs[i][1] *= 2
            ldiv!(T(2), PreCoeffs[i])
        end
    end

    PreCoeffs
end

function alias(uk::Vector{T}, bw::Integer) where {T}
    # alias the coefficients for preconditioner
    lenuk = length(uk)
    if lenuk > bw + 1
        ukalias = uk[1:bw+1]
        aliaswidth = 2 * (bw + 1)
        lenuk = length(uk)

        # aliasing on first Chebyshev points 
        # Mason, J. C. and Handscomb, D. C., Chebyshev polynomials, Chapman & Hall/CRC, Boca Raton, FL, 2003.  (pp. 153)
        ukalias[1] -= foldr(-, view(uk, aliaswidth+1:aliaswidth:lenuk); init=zero(T))
        for j = 2:bw+1
            ukalias[j] -= foldr(-, view(uk, aliaswidth-j+2:aliaswidth:lenuk); init=zero(T))
            ukalias[j] -= foldr(-, view(uk, aliaswidth+j:aliaswidth:lenuk); init=zero(T))
        end
    else
        # coefficient is short enough for preconditioner
        ukalias = copy(uk)
    end

    ukalias
end

function minL2(JacCoeffs::Vector{Vector{T}}, p::Integer) where {T}
    # get the best low-degree L^2 approximation of Jacobian coefficients
    # NOTE: p may be less than sqrt(log2(n)) since the lengths of coefficients in the first Newton iteration are often short.
    # NOTE: coefficients are divided by 2 except for the first element which is convenient for construction of preconditioner
    PreCoeffs = similar(JacCoeffs)

    # get the coefficient matrix
    A = minL2coeffs(maximum(length.(JacCoeffs)), p+length(JacCoeffs), T)
    # aliasing
    for i in eachindex(JacCoeffs)
        PreCoeffs[i] = minL2(JacCoeffs[i], p + i - 1, A)

        # divide for preconditioner construction
        if !isempty(PreCoeffs[i])
            PreCoeffs[i][1] *= 2
            ldiv!(T(2), PreCoeffs[i])
        end
    end
    PreCoeffs, p
end

function minL2coeffs(lenuk::Integer, bw::Integer, T)
    # construction of coefficient matrix used in best L^2 approximation
    # Note: the matrix is Toeplitz-plus-Hankel-plus-rank-one
    v = Vector{T}(undef, bw+lenuk)
    v[2:2:end] .= zero(T)
    map!(x->2/(1-x^2), view(v, 1:2:bw+lenuk), 0:2:bw+lenuk-1)
    A = Toeplitz(view(v, 1:bw+1), view(v, 1:lenuk)) + Hankel(view(v, 1:bw+1), view(v, bw+1:bw+lenuk))
    ldiv!(T(2), view(A, 3:2:bw+1, 1))

    A
end

function minL2(uk::Vector{T}, bw::Integer, A::AbstractMatrix{T}) where T
    # get the best m-degree approximation in the sense of L^2 norm
    # Note: the gradient at ukL2 of functional ||uk - ukL2||_{L^2} is zero
    # Note: the coefficient matrix is large enough, i.e., size(A, 1) >= m+1 and size(A, 2) >= length(uk)
    
    lenuk = length(uk)
    if lenuk > bw+1
        # construction of rhs
        b = view(A, 1:bw+1, bw+2:lenuk) * view(uk, bw+2:lenuk)
        # solve
        ukL2 = view(A, 1:bw+1, 1:bw+1) \ b
        axpy!(true, view(uk, 1:bw+1), ukL2)
    else
        # coefficient is short enough for preconditioner
        ukL2 = copy(uk)
    end

    ukL2
end


# full matrix of ultraspherical method
function matrix(JacCoeffs::Vector{Vector{T}}, bc::Matrix{T}) where T
    # Convert JacCoeffs to matrix of differential operator with boundary conditions bc
    # Note that bc are stored vertically, i.e., every column of bc represents a boundary condition 
    n, N = size(bc)
    A = Matrix{T}(undef, n, n)

    # assign boundary conditions
    tbcinds = CartesianIndices(transpose(bc))
    copyto!(A, tbcinds, transpose(bc), tbcinds)

    # assign operator
    L = quasi2diffmat(JacCoeffs, n)
    copyto!(view(A, N+1:n, 1:n), L)

    A
end

# matrix related to differential equation (except the boundary conditions)
function quasi2diffmat(JacCoeffs::Vector{Vector{T}}, n::Integer) where T
    # L = quasi2diffmat(JacCoeffs) returns a matrix L of the US representation of the differential operator 
    # L*u = (JacCoeffs[N+1]*D^{N} + ... + JacCoeffs[2]*D^{1} + JacCoeffs[1]*D^{0})*u.
    N = length(JacCoeffs) - 1

    # the 0th order term
    if isempty(JacCoeffs[1])
        L = zeros(T, n-N, n)
    else
        L = spconvertmat(n-N, n, 0, N, T) * multmat(n, 0, JacCoeffs[1])
    end

    for i = 1:N
        if length(JacCoeffs[i+1]) == 1
            if !iszero(JacCoeffs[i+1][1])
                # nonzero constant term
                axpy!(JacCoeffs[i+1][1], spconvertmat(n-N, n, i, N, T)*spdiffmat(n, n, i, T), L)
            end
        elseif !isempty(JacCoeffs[i+1])
            axpy!(true, spconvertmat(n-N, n, i, N, T)*multmat(n, i, JacCoeffs[i+1])*spdiffmat(n, n, i, T), L)
        end
    end

    L
end