using FFTW
# Contains code that is based in part on Chebfun v5's chebfun/standardChop, 
# chebfun/@chebtech2/vals2coeffs.m and chebfun/@chebtech2/coeffs2vals.m
# which is distributed with the following license:

# Copyright (c) 2015, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function standardChop(coeffs::AbstractVector{T}, tol = eps(T)) where T
    #  Reduce the number of coefficients by dropping the tail that can be discarded.
    #  See J. L. Aurentz and L. N. Trefethen, "Chopping as
    #  Chebyshev series", http://arxiv.org/abs/1512.01803, December 2015.

    #  Check magnitude of TOL:
    if tol >= 1
        return 1
    end

    #  Make sure COEFFS has length at least 16:
    n = length(coeffs)
    cutoff = n
    if  n < 17
        # resort to naive chop
        mx = maximum(abs, coeffs)
        if mx == 0
            return 0
        end
        for k=n:-1:1
            if abs(coeffs[k]) > tol*mx
                return k
            end
        end
        return 0
    end

    #  Step 1: Convert COEFFS to a new monotonically nonincreasing
    #          vector ENVELOPE normalized to begin with the value 1.

    envelope = reverse(abs.(coeffs))
    accumulate!(max, envelope, envelope)
    reverse!(envelope)
    if envelope[1] == 0
        cutoff = 1
        return cutoff
    else
        envelope = envelope ./ envelope[1]
    end

    #  Step 2: Scan ENVELOPE for a value PLATEAUPOINT, the first point J-1, if any,
    #  that is followed by a plateau.  A plateau is a stretch of coefficients
    #  ENVELOPE(J),...,ENVELOPE(J2), J2 = round(1.25*J+5) <= N, with the property
    #  that ENVELOPE(J2)/ENVELOPE(J) > R.  The number R ranges from R = 0 if
    #  ENVELOPE(J) = TOL up to R = 1 if ENVELOPE(J) = TOL^(2/3).  Thus a potential
    #  plateau whose starting value is ENVELOPE(J) ~ TOL^(2/3) has to be perfectly
    #  flat to count, whereas with ENVELOPE(J) ~ TOL it doesn't have to be flat at
    #  all.  If a plateau point is found, then we know we are going to chop the
    #  vector, but the precise chopping point CUTOFF still remains to be determined
    #  in Step 3.

    plateauPoint = 0
    for j = 2:n
        j2 = round(Int, 1.25*j + 5)
        if j2 > n
            #  there is no plateau: exit
            return cutoff
        end
        e1 = envelope[j]
        e2 = envelope[j2]
        r = 3*(1 - log(e1)/log(tol))
        plateau = (e1 == 0) | (e2/e1 > r)
        if plateau
            #  a plateau has been found: go to Step 3
            plateauPoint = j - 1
            break
        end
    end

    #  Step 3: fix CUTOFF at a point where ENVELOPE, plus a linear function
    #  included to bias the result towards the left end, is minimal.
    #
    #  Some explanation is needed here.  One might imagine that if a plateau is
    #  found, then one should simply set CUTOFF = PLATEAUPOINT and be done, without
    #  the need for a Step 3. However, sometimes CUTOFF should be smaller or larger
    #  than PLATEAUPOINT, and that is what Step 3 achieves.
    #
    #  CUTOFF should be smaller than PLATEAUPOINT if the last few coefficients made
    #  negligible improvement but just managed to bring the vector ENVELOPE below the
    #  level TOL^(2/3), above which no plateau will ever be detected.  This part of
    #  the code is important for avoiding situations where a coefficient vector is
    #  chopped at a point that looks "obviously wrong" with PLOTCOEFFS.
    #
    #  CUTOFF should be larger than PLATEAUPOINT if, although a plateau has been
    #  found, one can nevertheless reduce the amplitude of the coefficients a good
    #  deal further by taking more of them.  This will happen most often when a
    #  plateau is detected at an amplitude close to TOL, because in this case, the
    #  "plateau" need not be very flat.  This part of the code is important to
    #  getting an extra digit or two beyond the minimal prescribed accuracy when it
    #  is easy to do so.

    if plateauPoint != 0 && envelope[plateauPoint] == 0
        cutoff = plateauPoint
    else
        j3 = sum(envelope .>= tol^(7/6))
        if j3 < j2
            j2 = j3 + 1
            envelope[j2] = tol^(7/6)
        end
        @views cc = log10.(envelope[1:j2])
        axpy!(true, range(0, (-1/3)*log10(tol), length = j2), cc)
        d = argmin(cc)
        cutoff = max(d - 1, 1)
    end

    return cutoff
end

function quadwts(n::Integer, T = Float64)
    #  N weights for Clenshaw-Curtis quadrature on 2nd-kind Chebyshev points.

    if ( n == 0 )                     #  Spcial case (no points!)
        return zeros(T, 0)
    elseif ( n == 1 )                #  Special case (single point)
        return lmul!(2, ones(T, 1))
    else                                   #  General case
        c = Vector{T}(0:2:n-1)
        map!(x -> 2/(1-x^2), c, c)
        append!(c, view(c, fld(n, 2):-1:2))

        w = ifft(c)                       #  Interior weights

        map!(real, c, w)
        push!(c, real(w[1])/2)
        c[1] /= 2                        #  Boundary weights
        return c
    end
end

function standardChop!(x::AbstractVector{T}, tol = eps(T)) where T
    # delete the redundant elements in an coefficient
    m = standardChop(x, T(tol))
    deleteat!(x, m+1:length(x))
end


function coeffs2vals(coeffs::AbstractVector{T}) where T<:Number
    # Convert Chebyshev coefficients to values at Chebyshev points of the 2nd kind.

    #  Get the length of the input:
    n = length(coeffs)

    #  Trivial case (constant or empty):
    if ( n <= 1 )
        return copy(coeffs)
    end

    #  check for symmetry
    isEven = mapreduce(abs, max, view(coeffs, 2:2:n)) == 0
    isOdd = mapreduce(abs, max, view(coeffs, 1:2:n)) == 0

    # Mirror the coefficients (to fake a DCT using an FFT):
    tmp = reverse(coeffs)
    pop!(tmp); popfirst!(tmp)
    prepend!(tmp, coeffs)
    ldiv!(2, tmp)
    tmp[1], tmp[n] = coeffs[1], coeffs[n]

    if isreal(coeffs)
        #  Real-valued case:
        values = real(fft(tmp))
    elseif isreal(1im*coeffs)
        #  Imaginary-valued case:
        values = 1im*real(fft(imag(tmp)))
    else
        #  General case:
        values = fft(tmp)
    end

    #  Flip and truncate:
    deleteat!(values, n+1:2*n-2)
    reverse!(values)

    #  enforce symmetry
    if isEven
        axpy!(true, reverse(values), values)
        ldiv!(2, values)
    end

    if isOdd
        axpy!(-1, reverse(values), values)
        ldiv!(2, values)
    end

    return values
end

function coeffs2vals!(coeffs::AbstractVector{T}) where T<:Number
    # Convert Chebyshev coefficients to values at Chebyshev points of the 2nd kind.

    #  Get the length of the input:
    n = length(coeffs)

    #  Trivial case (constant or empty):
    if n <= 1 
        return coeffs
    end

    ldiv!(2, view(coeffs, 2:n-1))
    # Mirror the coefficients (to fake a DCT using an FFT):
    if isreal(coeffs)
        #  Real-valued case:
        # Mirror the coefficients (to fake a DCT using an FFT):
        tmp = reverse(coeffs)
        pop!(tmp)
        popfirst!(tmp)
        prepend!(tmp, coeffs)

        fft_temp = fft(tmp)

        #  Flip and truncate:
        map!(real, coeffs, view(fft_temp, n:-1:1))
    else
        #  General case:
        append!(coeffs, view(coeffs, n-1:-1:2))
        fft!(coeffs)

        #  Flip and truncate:
        deleteat!(coeffs, n+1:2*n-2)
        reverse!(coeffs)
    end

    coeffs
end

function vals2coeffs(values::AbstractArray{T, 1}) where T
    # Convert values at Chebyshev points to Chebyshev coefficients.

    #  Get the length of the input:
    n = length(values)

    #  Trivial case (constant):
    if ( n <= 1 )
        return copy(values)
    end

    #  check for symmetry
    isEven = maximum(abs.(values-reverse(values))) == 0
    isOdd = maximum(abs.(values+reverse(values))) == 0

    #  Mirror the values (to fake a DCT using an FFT):
    tmp = reverse(values)
    pop!(tmp)
    append!(tmp, values)
    pop!(tmp)

    if isreal(values)
        #  Real-valued case:
        coeffs = real(ifft(tmp))
    elseif isreal(1im*values)
        #  Imaginary-valued case:
        coeffs = ifft(imag(tmp))
        coeffs = 1im .* real(coeffs)
    else
        #  General case:
        coeffs = ifft(tmp)
    end

    #  Truncate:
    deleteat!(coeffs, n+1:2*n-2)

    #  Scale the interior coefficients:
    lmul!(2, view(coeffs, 2:n-1))

    #  adjust coefficients for symmetry
    if isEven
        coeffs[2:2:end] .= 0
    end
    if isOdd
        coeffs[1:2:end] .= 0
    end

    return coeffs
end

function vals2coeffs!(values::AbstractArray{T, 1}) where T
    # Convert values at Chebyshev points to Chebyshev coefficients.

    #  Get the length of the input:
    n = length(values)

    #  Trivial case (constant):
    if ( n <= 1 )
        return
    end

    if isreal(values)
        #  Real-valued case:
        #  Mirror the values (to fake a DCT using an FFT):
        tmp = reverse(values)
        pop!(tmp)
        append!(tmp, values)
        pop!(tmp)

        ifft_temp = ifft(tmp)

        #  Truncate:
        map!(real, values, view(ifft_temp, 1:n))
    else
        #  General case:
        #  Mirror the values (to fake a DCT using an FFT):
        tmp = reverse(values)
        pop!(tmp)
        prepend!(values, tmp)
        pop!(values)

        ifft!(values)

        #  Truncate:
        deleteat!(values, n+1:2*n-2)
    end

    #  Scale the interior coefficients:
    lmul!(2, view(values, 2:n-1))

    values
end


function chebpts(n::Integer, T = Float64)
    if n == 0
        return zeros(T, 0)
    elseif n == 1
        return zeros(T, 1)
    else
        m = n - 1
        x = lmul!(pi/2m, Vector{T}(-m:2:m))  # (Use of sine enforces symmetry.)
        map!(sin, x, x)
        return x
    end
end

function myop(op::Function, x::AbstractVector{T}) where T<:AbstractFloat
    if isempty(x)
        return zeros(T, 0)
    end
    
    n = max(nextpow(2, length(x)), 32) + 1 # fft works on 2*n-2
    lenx = length(x)
    opx = Vector{T}(undef, n)
    copyto!(opx, x)
    opx[lenx+1:n] .= 0
    maxlength = 2^17 + 1
    while n < maxlength
        # evaluate the prolonged terms by sin function
        coeffs2vals!(opx)
        map!(op, opx, opx)
        vals2coeffs!(opx)
        m = standardChop(opx)

        if m < n
            deleteat!(opx, m+1:n)
            return opx
        else
            # reset opx and prolong it
            opx[lenx+1:n] .= 0
            copyto!(opx, x)
            append!(opx, zeros(T, n-1))
            n = 2*n - 1
        end
    end

    error(string(string(op), "(u) may not be exact enough"))
    return opx
end

for op in (:sin, :cos, :tan, :sinh, :cosh, :tanh, :exp, :log)
    mop = Symbol("my", op)
    @eval begin
        $mop(x::AbstractVector) = myop($op, x)
    end
end

function coeffs(f::Function, n::Integer, T = Float64)
    #  get the Chebyshev coefficients of function f on a (n+1) grids of 2nd-kind Chebyshev points.
    #  the input f is a function handle, n is the number of Chebyshev points
    #  which is enough for computing the Chebyshev coefficients of f

    x = chebpts(n+1, T)  # Chebyshev points
    map!(f, x, x)         # function application
    vals2coeffs!(x)
end