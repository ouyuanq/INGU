# fast evaluation of differentiation, multiplication and conversion for ultraspherical spectral method

## fast differentiation operator (sparse, one superdiagonal)
function fastdiff(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the differentiation matrix to vector a. The vector a is differentiated lambda times.
    @assert lambda >= 0 "Can not differentiate at negative order"
    lena = length(a)
    if lena > lambda
        if lambda > 0
            b = Vector{T}(undef, lena - lambda)
            con = T(2^(lambda-1)*factorial(lambda - 1))
            @inbounds for i in eachindex(b)
                lambdai = lambda + i
                b[i] = (lambdai - 1) * a[lambdai]
            end
            lmul!(con, b)
        else
            b = copy(a)
        end
    else
        b = zeros(T, 1)
    end

    return b
end

function fastdiff!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the differentiation matrix to vector a. The vector a is differentiated lambda times and the result is stored in a.
    @assert lambda >= 0 "Can not differentiate at negative order"
    lena = length(a)
    if lena > lambda
        if lambda > 0
            deleteat!(a, 1:lambda)
            con = T(2^(lambda-1)*factorial(lambda - 1))
            for i in eachindex(a)
                a[i] *= lambda+i-1
            end
            lmul!(con, a)
        end
    else
        a .= zero(T)
    end

    a
end

function fastdiff!(da::AbstractVector{T}, a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the differentiation matrix to vector a and store the result in da where da is assumed to be no shorter than a
    @assert lambda >= 0 "Can not differentiate at negative order"
    lena = length(a)
    deleteat!(da, lena+1:length(da))
    if lena > lambda
        con = T(2^(lambda-1)*factorial(lambda - 1))
        for i = 1:lena-lambda
            lambdai = lambda + i
            da[i] = con * (lambdai - 1) * a[lambdai] 
        end
        da[lena-lambda+1:lena] .= zero(T)
    else
        da .= zero(T)
    end
    
    da
end

## fast conversion operator (sparse, diagonal and one superdiagonal)
function fastconv!(a::AbstractVector{T}, mu::Integer, lambda::Integer) where T<:Number
    # convert C^{mu} series a to C^{lambda} basis
    # The result is stored in the initial vector
    for i = mu:lambda-1
        fastconv1!(a, i)
    end
    a
end

function fastconv1!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the conversion matrix S^{lambda} to vector a. The result is stored in the initial vector
    n = length(a)

    if n > 2   # the usual case
        if lambda == 0
            ldiv!(T(2), a)
            a[1] *= 2
        else
            lmul!(lambda, a)
            den = lambda
            for i in eachindex(a)
                a[i] /= den
                den += 1
            end
        end
        axpy!(-one(T), view(a, 3:n), view(a, 1:n-2))  # the second superdiagonal
    elseif n == 2
        if lambda == 0
            a[2] = a[2]/2
        else
            a[2] = a[2]*lambda/(lambda + 1)
        end
    end
end

## fast inversion operator (semiseparable and can be applied by backward substitution)
function fastinve!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # convert a C^{lambda} series to a Chebyshev T series. The result is stored in the initial vector.
    if lambda < 0
        error("Lambda should be larger than 0 in function fastinve")
    end

    for i = lambda-1:-1:0
        fastinve1!(a, i)
    end

    a
end

function fastinve2U!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # convert a C^{lambda} series to a Chebyshev U series. The result is stored in the initial vector.
    if lambda < 1
        error("Lambda should be larger than 1 in function fastinve2U")
    end

    for i = lambda-1:-1:1
        fastinve1!(a, i)
    end

    a
end

function fastinve1!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the inverse of the conversion matrix S_{lambda} to vector a. The result is stored in the initial vector
    n = length(a)

    if n > 2    # the usual case
        @inbounds for i = n:-1:3
            a[i-2] += a[i]
        end
        if lambda == 0
            lmul!(T(2), a)
            a[1] /= T(2)
        else
            ldiv!(lambda, a)
            lambda1 = lambda - 1
            @inbounds for i in eachindex(a)
                a[i] *= lambda1 + i
            end
        end
    elseif n == 2
        if lambda == 0
            a[2] *= 2
        else
            a[2] *= ((lambda + 1)/lambda)
        end
    end

    a
end

## fast differentiation to Chebyshev T series
# dive = di(fferentiation) + (in)ve(rsion)
function fastdive(u::AbstractVector{T}, lambda::Integer) where T
    # apply differentiation matrix D_{lambda} to u and convert it to a Chebyshev T series

    v = fastdiff(u, lambda)
    fastinve!(v, lambda)

    v
end

## fast differentiation to Chebyshev U series (which is necessary in our fast multiplication)
# dive = diff + inve(rse)
function fastdive2U(u::AbstractVector{T}, lambda::Integer) where T
    # apply differentiation matrix D_{lambda} to u and convert it to a Chebyshev U series
    @assert lambda >= 1 "Lambda should be larger than 1 in function fastinve2U"

    v = fastdiff(u, lambda)
    fastinve2U!(v, lambda)
end

function fastdive2U!(du::AbstractVector{T}, u::AbstractVector{T}, lambda::Integer) where T
    # apply differentiation matrix D_{lambda} to u and convert it to a Chebyshev U series
    # the result is stored in du and du is assumed to be no shorter than u
    @assert lambda >= 1 "Lambda should be larger than 1 in function fastinve2U"

    fastdiff!(du, u, lambda)
    fastinve2U!(du, lambda)
end

## fast multiplication operator (Toeplitz + Hankel where FFT can be used)
function fastmult!(c::Number, fc::AbstractVector{T}, gc::AbstractVector{U}) where {T, U}
    rmul!(fc, c)
    fastmult!(fc, gc)
end

function fastmult!(fc::AbstractVector{T}, gc::AbstractVector{U}) where {T, U}
    # compute the coefficient fc * gc where fc is a Chebyshev T series and gc is a Chebyshev T series the result is a Chebyshev T series
    # Note: fc and gc can be swapped since they are in the same basis. There is no restrictions on the lengths of fc or gc.
    # This function is equivalent to compute the matrix-vector product M₀[gc]*fc, where M₀[·] denotes multiplication operator in Chebyshev T basis and the result is also a Chebyshev T series
    lf, lg = length(fc), length(gc)

    if lf == 1
        append!(fc, gc)
        rmul!(fc, fc[1])
        popfirst!(fc)
    elseif lg == 1
        rmul!(fc, gc[1])
    elseif lg == 2
        # Toeplitz part only
        temp = (gc[2]/2) .* fc
        lmul!(gc[1], fc)
        push!(fc, zero(T))

        # superdiagonal and subdiagonal of Toeplitz matirx
        @views begin
            axpy!(true, temp[2:end], fc[1:lf-1])
            axpy!(true, temp, fc[2:end])
        end

        # one element from Hankel part
        fc[2] += temp[1]
    else
        coeff_timesT!(fc, gc, lf+lg-1)
    end

    fc
end

function coeff_timesT!(fc::AbstractVector{T}, gc::AbstractVector{U}, lfg::Integer) where {T, U}
    #   returns the vector of Chebyshev coefficients resulting from the multiplication of two functions with FC and GC
    #   coefficients. The vectors have already been prolonged.

    #   Multiplication in coefficient space is a Toeplitz-plus-Hankel-plus-rank-one
    #   operator (see Olver & Townsend, A fast and well-conditioned spectral method,
    #   SIAM Review, 2013). This can be embedded into a Circular matrix and applied
    #   using the FFT.

    lf, lg = length(fc), length(gc)
    W = promote_type(T, U)
    temp = Vector{W}(undef, 2*lfg - 1)
    copyto!(temp, fc)
    temp[lf+1:2*lfg-lf] .= 0
    copyto!(view(temp, 2*lfg+1-lf:2*lfg-1), view(fc, lf:-1:2))
    temp[1] *= 2

    plan = plan_fft(temp)
    fc_fft = plan * temp

    copyto!(temp, gc)
    temp[lg+1:2*lfg-lg] .= 0
    copyto!(view(temp, 2*lfg+1-lg:2*lfg-1), view(gc, lg:-1:2))
    temp[1] *= 2

    gc_fft = plan * temp

    broadcast!(*, fc_fft, fc_fft, gc_fft) # dot multiplication
    x = ifft(fc_fft)

    append!(fc, Vector{T}(undef, lg-1))
    map!(real, fc, view(x, 1:lfg))
    ldiv!(2, fc)
    fc[1] /= 2
end

# square of a Chebyshev series (i.e., fc and gc are the same in fastmult)
function fastsquare!(fc::AbstractVector{T}) where T
   # compute the coefficient fc * gc where both fc and gc are Chebyshev T series
    # plan and plani can be used to accelerate the computation of FFT and iFFT
    lf = length(fc)

    if lf == 1
        fc[1] = fc[1]^2
    else
        lf2 = 2*lf - 1
        coeff_timesT!(fc, lf2, lf)
    end

    standardChop!(fc)
end

function coeff_timesT!(fc::AbstractVector{T}, lf2::Integer, lf::Integer) where {T}
    #   returns the vector of Chebyshev coefficients of the square of fc. The vector has already been prolonged.

    #   Multiplication in coefficient space is a Toeplitz-plus-Hankel-plus-rank-one
    #   operator (see Olver & Townsend, A fast and well-conditioned spectral method,
    #   SIAM Review, 2013). This can be embedded into a Circular matrix and applied
    #   using the FFT.

    x = Vector{T}(undef, 2*lf2 - 1)
    copyto!(x, fc)
    x[lf+1:3*lf-2] .= 0
    @views x[3*lf-1:end] .= fc[lf:-1:2]
    x[1] *= 2

    fft_x = fft(x)
    broadcast!(*, fft_x, fft_x, fft_x)
    x = ifft(fft_x)

    append!(fc, Vector{T}(undef, lf-1))
    map!(real, fc, view(x, 1:lf2))
    ldiv!(2, fc)
    fc[1] /= 2
end

function fastsquare!(fc::AbstractVector{T}, Plan, Plani) where {T}
    # compute the coefficient fc * gc where both fc and gc are Chebyshev T series
    # plan and plani can be used to accelerate the computation of FFT and iFFT
    lf = length(fc)

    if lf == 1
        fc[1] = fc[1]^2
    else
        x = Vector{Complex{T}}(undef, 2 * lf - 1)
        copyto!(x, fc)
        x[lf+1:3*lf-2] .= 0
        copyto!(view(x, 3*lf-1:2*lf2-1), view(fc, lf:-1:2))
        x[1] *= 2

        Plan * x
        broadcast!(*, x, x, x)
        Plani * x

        append!(fc, Vector{T}(undef, lf - 1))
        map!(real, fc, x)
        ldiv!(2, fc)

        fc[1] *= T(0.5)
    end

    standardChop!(fc)
end

# multiplication of different basis
function fastmult1!(c::Number, fc::AbstractVector{T}, gc::AbstractVector{T}) where {T}
    rmul!(fc, c)
    fastmult1!(fc, gc)
end

function fastmult1!(fc::AbstractVector{T}, gc::AbstractVector{T}) where {T}
    # compute the coefficients of product of fc and gc, where fc is a Chebyshev U series and gc is a Chebyshev T series. This function is equivalent to compute the matrix-vector product M₁[gc]*fc, where M₁[·] denotes multiplication operator in C^{1} basis and the result is based on C^{1} basis.
    lf, lg = length(fc), length(gc)

    if lf == 1
        append!(fc, gc)
        rmul!(fc, fc[1])
        popfirst!(fc)
        fastconv!(fc, 0, 1) # gc is a Chebyshev T series
    elseif lg == 1
        rmul!(fc, gc[1])
    elseif lg == 2
        # Toeplitz part only
        temp = (gc[2]/2) .* fc
        lmul!(gc[1], fc)
        push!(fc, zero(T))

        # superdiagonal and subdiagonal of Toeplitz matirx
        @views begin
            axpy!(true, temp[2:end], fc[1:lf-1])
            axpy!(true, temp, fc[2:end])
        end
    else
        coeff_timesU!(fc, gc, lf+lg-1, lf, lg)
    end

    fc
end

function coeff_timesU!(fc::AbstractVector{T}, gc::AbstractVector{T}, lfg::Integer, lf::Integer, lg::Integer) where {T}
    #COEFF_TIMESU!(FC, GC)   Multiplication in coefficient space
    #   HC = COEFF_TIMES(FC, GC) returns the vector of Chebyshev U coefficients, HC,
    #   resulting from the multiplication of two functions with FC and GC
    #   coefficients, where FC is a Chebyshev U series and GC is a Chebyshev T
    #   series. The vectors have already been prolonged.

    #   Multiplication in coefficient space is a Toeplitz-plus-Hankel operator.
    #   This can be embedded into a Circular matrix and applied using the FFT.

    #   lfg is the length of both coefficients and lf and lg are the length of nonzero parts of fc and gc.

    temp = Vector{T}(undef, 2*lfg - 1)
    copyto!(temp, fc)
    temp[lf+1:2*lfg-lf-2] .= 0
    axpby!(-1, view(fc, lf:-1:1), 0, view(temp, 2*lfg-1-lf:2*lfg-2))
    temp[end] = 0

    plan = plan_fft(temp)
    fc_fft = plan * temp

    copyto!(temp, gc)
    temp[lg+1:2*lfg-lg] .= 0
    copyto!(view(temp, 2*lfg+1-lg:2*lfg-1), view(gc, lg:-1:2))
    temp[1] *= 2

    gc_fft = plan * temp

    broadcast!(*, fc_fft, fc_fft, gc_fft) # dot multiplication
    x = ifft(fc_fft)

    # correction for the last two elements. NOTE: there are only one and two nonzero products in the Hankel part computation of last two elements respectively.
    x[lfg - 1] += fc[lf] * gc[lg]
    x[lfg] += fc[lf] * gc[lg - 1] + fc[lf - 1] * gc[lg]

    append!(fc, Vector{T}(undef, lg-1))
    map!(real, fc, view(x, 1:lfg))
    ldiv!(2, fc)
end


# fewer FFTs by factoring out inverse FFT
function fastmult1_eco!(complex_g::AbstractVector{Complex{T}}, complex_delta::AbstractVector{Complex{T}}, complex_coeff::AbstractVector{Complex{T}}, delta::AbstractVector{T}, coeff::AbstractVector{T}, Plan) where T
    # the first half part of fast Toeplitz and Hankel matrices application, i.e., element-wise multiplication of FFT transformed vectors
    # the results are added up in complex_g before a single inverse FFT is applied
    # note that length(coeff) > 2 is assured
    n_fft = length(complex_delta)
    L = cld(length(Plan), 2)
    lendelta, lencoeff = length(delta), min(length(coeff), L)

    # assignment for Toeplitz part
    copyto!(complex_delta, delta)
    complex_delta[lendelta+1:end] .= 0

    copyto!(view(complex_coeff, 1:lencoeff), view(coeff, 1:lencoeff))
    complex_coeff[1] *= 2
    complex_coeff[lencoeff+1:end-lencoeff+1] .= 0
    copyto!(view(complex_coeff, n_fft-lencoeff+2:n_fft), view(coeff, lencoeff:-1:2))

    # multiplication of Toeplitz part
    Plan * complex_delta
    Plan * complex_coeff
    broadcast!(*, complex_delta, complex_delta, complex_coeff)
    axpy!(true, complex_delta, complex_g)

    # assignment for Hankel part
    # note that elements in Hankel part may be more than those in Toeplitz part
    complex_delta[1:L-lendelta] .= 0
    copyto!(view(complex_delta, L-lendelta+1:L), view(delta, lendelta:-1:1))
    complex_delta[L+1:end] .= 0

    copyto!(view(complex_coeff, L+2:L+lencoeff-1), view(coeff, 3:lencoeff))
    if length(coeff) > L
        complex_coeff[end] = coeff[L+1]
        if length(coeff) > L + 1
            copyto!(view(complex_coeff, 1:min(L, length(coeff)-L-1)), view(coeff, L+2:min(2*L+1, length(coeff))))
        end
        complex_coeff[min(L, length(coeff)-L-1)+1:L+1] .= 0
    else
        complex_coeff[L+lencoeff:end] .= 0
        complex_coeff[1:L+1] .= 0
    end

    # multiplication of Hankel part
    Plan * complex_delta
    Plan * complex_coeff
    broadcast!(*, complex_delta, complex_delta, complex_coeff)
    axpy!(-1, complex_delta, complex_g) # note the minus sign!

    complex_g
end

function fastmult1_eco!(complex_g::AbstractVector{Complex{T}}, complex_deltaT::AbstractVector{Complex{T}},complex_deltaH::AbstractVector{Complex{T}}, complex_coeff::AbstractVector{Complex{T}}, coeff::AbstractVector{T}, Plan) where T
    # the first half part of fast Toeplitz and Hankel matrices application, i.e., element-wise multiplication of FFT transformed vectors
    # the FFT transformations related to delta for both Toeplitz and Hankel matrices are done already
    # note that length(coeff) > 2 is assured
    n_fft = length(complex_deltaT)
    L = cld(length(Plan), 2)
    lencoeff = min(length(coeff), L)

    # assignment for Toeplitz part
    copyto!(view(complex_coeff, 1:lencoeff), view(coeff, 1:lencoeff))
    complex_coeff[1] *= 2
    complex_coeff[lencoeff+1:end-lencoeff+1] .= 0
    copyto!(view(complex_coeff, n_fft-lencoeff+2:n_fft), view(coeff, lencoeff:-1:2))

    # multiplication of Toeplitz part
    Plan * complex_coeff
    broadcast!(*, complex_coeff, complex_deltaT, complex_coeff)
    axpy!(true, complex_coeff, complex_g)

    # assignment for Hankel part
    # note that elements in Hankel part may be more than those in Toeplitz part
    copyto!(view(complex_coeff, L+2:L+lencoeff-1), view(coeff, 3:lencoeff))
    if length(coeff) > L
        complex_coeff[end] = coeff[L+1]
        if length(coeff) > L + 1
            copyto!(view(complex_coeff, 1:min(L, length(coeff)-L-1)), view(coeff, L+2:min(2*L+1, length(coeff))))
        end
        complex_coeff[min(L, length(coeff)-L-1)+1:L+1] .= 0
    else
        complex_coeff[L+lencoeff:end] .= 0
        complex_coeff[1:L+1] .= 0
    end

    # multiplication of Hankel part
    Plan * complex_coeff
    broadcast!(*, complex_coeff, complex_deltaH, complex_coeff)
    axpy!(-1, complex_coeff, complex_g) # note the minus sign!

    complex_g
end

function fastmult1_eco!(gtemp::AbstractVector{T}, complex_g::AbstractVector{Complex{T}}, Plani) where T
    # apply the inverse FFT to complex_g and add the result to gtemp
    Plani * complex_g
    L = cld(length(Plani), 2)

    axpy!(T(0.5), real(view(complex_g, 1:L)), view(gtemp, 1:L))

    gtemp
end

## intergal operator
function fastquad(x::AbstractVector{T}) where T
    # appply Clenshaw–Curtis quadrature formula to compute the quadrature of a Chebyshev series
    c = zero(T)
    for i = 1:2:length(x)
        c += 2 * x[i] / (1 - (i-1)^2)
    end

    c
end

## fast methods related to applying the transpose of operators (differentiation, multiplication, conversion and inversion)

## transpose of conversion operator (two nonzero diagonals)
function fastconv_trans(a::AbstractVector, lambda::Integer, ::Type{T}) where T<:AbstractFloat
    # apply the transpose of conversion matrix to vector a.
    # NOTE: the result vector b is 2 elements longer than a since S_{\lambda}^{-1} has a nonzero second subdiagonal
    n = length(a)
    b = Vector{T}(undef, n+2) # the output
    b[n+1:n+2] .= zero(T) # the extra two elements
    copyto!(b, a)  # the diagonal
    axpy!(-one(T), a, view(b, 3:n+2))  # the second subdiagonal

    if lambda == 0
        ldiv!(2, view(b, 2:n+2))
    else
        lmul!(lambda, b)
        @inbounds for i = 0:n+1
            b[i+1] /= (lambda+i) 
        end
    end

    b
end

function fastconv_trans!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the transpose of conversion matrix to vector a and the result is stored in a.
    # Note that the conversion matrix is sparse.
    n = length(a)

    if n == 0
        return a
    end
    append!(a, zeros(T, 2))

    # the second subdiagonal
    @inbounds for i = n:-1:1
        a[i+2] -= a[i]
    end

    if lambda == 0
        ldiv!(2, view(a, 2:n+2))
    else
        lmul!(lambda, a)
        @inbounds for i = 0:n+1
            a[i+1] /= (lambda+i) 
        end
    end

    a
end


function fastconv_trans(a::AbstractVector, mu::Integer, lambda::Integer, ::Type{T}) where T<:AbstractFloat
    # apply the transpose of conversion matrix which convert C^{mu} series a to C^{lambda} basis
    if lambda == mu
        b = copy(a)
    else
        b = fastconv_trans(a, lambda-1, T)
        @inbounds for i = lambda-2:-1:mu
            fastconv_trans!(b, i)
        end
    end

    b
end

## transpose of inversion operator (off-diagonal low rank)
function fastinve1_trans!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the transpose of inverse of the conversion matrix C^{(lambda)} to vector a
    # Forward substitution is used and the result is stored in the initial vector.
    n = length(a)

    if lambda == 0
        a[2] *= 2

        @inbounds for i = 3:n
            a[i] = a[i-2] + 2*a[i]
        end
    else
        numerator = lambda + 1
        a[2] *= (numerator/lambda)
        numerator += 1

        @inbounds for i = 3:n
            a[i] = a[i-2] + a[i] * (numerator/lambda)
            numerator += 1
        end
    end

    a
end

function fastinve_trans!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # convert a C^{lambda} series to a Chebyshev T series. The result is stored in the initial vector
    @inbounds for i = 0:lambda-1
        fastinve1_trans!(a, i)
    end

    a
end

function fastinve2U_trans!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # convert a C^{lambda} series to a Chebyshev U series. The result is stored in the initial vector.
    if lambda < 1
        error("Lambda should be larger than 1 in function finve2U")
    end

    @inbounds for i = 1:lambda-1
        fastinve1_trans!(a, i)
    end

    a
end


## transpose of differentiation operator (one nonzero diagonal)
function fastdiff_trans!(a::AbstractVector{T}, lambda::Integer) where T<:Number
    # apply the transpose of differetiation matirx D^{(lambda)} to vector a. 
    # The result is stored in the initial vector.
    @assert lambda >= 0 "Can not differentiate at negative order"

    if lambda > 0
        factor = lambda
        @inbounds for i in eachindex(a)
            a[i] *= factor
            factor += 1
        end

        con = 2^(lambda-1)*factorial(lambda - 1)
        lmul!(con, a)
        prepend!(a, zeros(T, lambda))
    end

    a
end


## transpose of fast differentiation to Chebyshev T series
# inff = in(version) + (di)ff(erentiation)
function fastinff!(u::AbstractVector, order::Integer)
    # transpose of inversion after differetiation
    fastinve_trans!(u, order)
    fastdiff_trans!(u, order)
end

function fastinff4U!(u::AbstractVector, order::Integer)
    # transpose of inversion to U basis after differetiation
    # apply the transpose of inversion and transpose of differentiation for U basis
    @assert order >= 1 "order should be no less than 1."
    fastinve2U_trans!(u, order)
    fastdiff_trans!(u, order)
end



# sparse ultraspherical matrices which are used as building bolcks for ultraspherical spectral method
## Conversion
function spconvertmat(m::Integer, n::Integer, mu::Integer, lambda::Integer, T::DataType)
    # construct a sparse convert matrix representing the conversion operator between two ultraspherical polynomials.
    # The resulting matrix converts coefficients in C^{mu} basis to C^{lambda} basis while keeping the length of the coefficients. m, n denotes the number of rows and columns of the matrix.
    # Note that we require mu is no more that lambda

    @assert mu <= lambda "μ should be no more than λ."

    if mu == lambda
        S = spdiagm(m, n, 0 => ones(T, min(m, n)))
    else
        S = spconvertmat_one(m, n, lambda - 1, T)
        for i = lambda-2:-1:mu
            S *= spconvertmat_one(n, n, i, T)
        end
    end

    S
end

function spconvertmat_one(m::Integer, n::Integer, mu::Integer, T::DataType)
    # compute the sparse matrix for converting coefficients in C^{mu} basis to C^{mu+1} basis. The dimension of the resulting matrix is m x n. 
    # Note that m is assumed to be no larger than n.
    if mu == 0
        a = lmul!(T(0.5), ones(T, min(m+2, n)))
        a[1] = 1
        S = spdiagm(m, n, 0 => view(a, 1:m), 2 => view(a, 3:min(m+2, n)))
    else
        a = Vector{T}(mu:mu+min(m+2, n)-1)
        for i in eachindex(a)
            a[i] = mu / a[i]
        end
        S = spdiagm(m, n, 0 => view(a, 1:m), 2 => view(a, 3:min(m+2, n)))
    end

    lmul!(-1, view(S, diagind(S, 2)))  # negative superdiagonal
    S
end

## Differetiation
function spdiffmat(m::Integer, n::Integer, lambda::Integer, T::DataType)
    # construct a sparse ultraspherical differetiation matrix which differetiate a Chebyshev T basis to a C^{lambda} basis
    if lambda == 0
        S = spdiagm(m, n, 0 => ones(T, min(m, n)))
    else
        a = lmul!(2^(lambda - 1)*factorial(lambda-1), convert(Vector{T}, (lambda:n-1)))
        S = spdiagm(m, n, lambda => a)
    end

    S
end

## Multiplication
function multmat(n::Integer, lambda::Integer, a::AbstractVector{T}) where T
    # construct a dense multiplication matrix which represents multiplying a C^{lambda} basis coefficients u to a C^{lambda} basis coefficients.
    # NOTE: We assume that u is based on Chebyshev T basis. The resulting maxtrix has dimension n × n

    la = length(a)
    # empty term
    if la == 0
        return zeros(T, n, n)
    end
    if la == 1
        # Multiplying by a scalar is easy
        M = diagm(n, n, lmul!(a[1], ones(T, n)))
        return M
    end

    if lambda == 0
        # factor out 1/2
        atemp = ldiv!(2, copy(a))
        atemp[1] = a[1]

        # Toeplitz and Hankel part
        if la < 2*n - 1
            # make atemp long enough for generators of Hankel matrices
            append!(atemp, zeros(T, 2*n-1-la))
        end
        ST = SymmetricToeplitz(view(atemp, 1:n))  # Toeplitz part
        H = Hankel(view(atemp, 1:2*n-1))  # Hankel part
        # Toeplitz-plus-Hankel-plus-rank 1
        M = Matrix(H)
        M[1, :] .= 0
        axpy!(true, ST, M)
    elseif lambda == 1
        # factor out 1/2
        atemp = ldiv!(2, copy(a))
        atemp[1] = a[1]

        # Toeplitz and Hankel part
        if la < 2*n + 1
            # make atemp long enough for generators of Hankel matrices
            append!(atemp, zeros(T, 2*n+1-la))
        end
        ST = SymmetricToeplitz(view(atemp, 1:n))  # Toeplitz part
        H = Hankel(view(atemp, 3:2*n+1))  # Hankel part
        # Toeplitz-plus-Hankel
        M = ST - H
    else
        # Prolong or truncate coefficients
        if la > n
            atemp = a[1:n]
        else
            atemp = append!(copy(a), zeros(T, n - la))
        end

        # There is no structure that we can exploit and recurrence are used here.
        fastconv!(atemp, 0, lambda)
        # a larger matrix is generated for exactness
        lsafe = 2n

        M0 = diagm(ones(T, lsafe))
        Mx = diagm(-1 => convert(Vector{T}, (1:lsafe-1) ./ (2lambda:2:2(lambda+lsafe-2))), 1 => convert(Vector{T}, (2lambda:2lambda+lsafe-2) ./ (2(lambda+1):2:2(lambda+lsafe-1))))
        M1 = lmul!(2lambda, Matrix(Mx))

        # Construct the multiplication operator by a three-term recurrence:
        M = lmul!(atemp[1], Matrix(M0))
        axpy!(atemp[2], M1, M)
        M2 = similar(M)
        for j = 3:n
            mul!(M2, Mx, M1, true, false)
            axpby!(-(j + 2*lambda - 3)/(j - 1), M0, 2*(j + lambda - 2)/(j - 1), M2)
            axpy!(atemp[j], M2, M)
            M0, M1, M2 = M1, M2, M0
            # Early break if coefficients are sufficiently small
            if maximum(abs, view(atemp, j+1:n)) < eps(T)
                break
            end
        end

        # Chopping M
        M = view(M, 1:n, 1:n)
    end

    M
end