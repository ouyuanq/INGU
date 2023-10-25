# construct the almost banded preconditioner for linear equations in nonlinear iterations
# NOTE: the almost banded matrix is NOT constructed EXACTLY since it is a preconditioner and some logics may not be suitable for usual almost banded matrices for ultraspherical spectral method
# the preconditioner is S_{N-1}...S_1(∑M_1[a^λ] S_1^{-1}...S_{λ-1}^{-1} D_{λ} + S_0 M_0[a^0]), where a^{λ} is obtained during the nonlinear iterations. Note that the expression in the parenthesis is an operator that maps a Chebyshev series into C^{(1)} series.
using SemiseparableMatrices, BandedMatrices

function bandinit(lincoeffs::Vector{Vector{T}}, n::Integer, p::Integer) where {T}
    # construct the initial band parts of almost banded preconditioner

    # initialization
    N = length(lincoeffs) - 1
    pN = p+N

    # bands to be determined (n-N rows in total)
    bands = zeros(T, n - N + 2 * (N - 1), 2 * pN + 1) # extra 2*(N-1) rows for conversion matrices
    szbands = size(bands)
    constantterm = zeros(Integer, 0)
    for i in eachindex(lincoeffs)
        if isempty(lincoeffs[i])
            # do nothing for empty coefficients
            continue
        end

        # Note that lincoeffs[i] is not empty here
        if length(lincoeffs[i]) > 1
            # terms with variable coefficients
            bandsgenerate!(bands, lincoeffs[i], 1, n - N, p, i-1, N)
        elseif i == 1
            # constant 0th order term (S_0 is multiplied)
            bands[1, p+1] += lincoeffs[i][1]
            bands[2:end, p+1] .+= lincoeffs[i][1] / 2
            bands[:, p+3] .-= lincoeffs[i][1] / 2
        elseif i == 2
            # constant 1st order term
            for j = 1:szbands[1]
                bands[j, p+2] += j * lincoeffs[i][1]
            end
        else
            # leave them along
            push!(constantterm, i - 1)
        end
    end
    # S_{N-1}...S_1(from right to left)
    for j = 1:N-1
        scale = j ./ (j:j+szbands[1]-1)
        broadcast!(*, bands, bands, scale)

        # add up because two nonzero diagonals
        @views bands[1:szbands[1]-2*j, 3:szbands[2]] .-= bands[3:szbands[1]-2*(j-1), 1:szbands[2]-2]
    end

    # higher order (>=2) terms with constant coefficients
    for i in constantterm
        if !iszero(lincoeffs[i+1][1])
            bandconst = zeros(T, n - N + 2 * (N - i), 2 * (N - i) + 1)
            szbandconst = size(bandconst)
            if i == 0
                bandconst[:, 1] .= one(T)
            else
                bandconst[:, 1] .= 2^(i - 1) * factorial(i - 1) .* ((1:szbandconst[1]) .+ (i - 1))
            end
            # S_{N-1}...S_i(from right to left)
            for j = i:N-1
                scale = j ./ (j:j+szbandconst[1]-1)
                broadcast!(*, bandconst, bandconst, scale)
                # bandconst .*= scale

                # add up because two nonzero diagonals
                @views bandconst[1:szbandconst[1]-2*(j-i+1), 3:szbandconst[2]] .-= bandconst[3:szbandconst[1]-2*(j-i), 1:szbands[2]-2]
            end

            # add up
            axpy!(lincoeffs[i+1][1], bandconst, view(bands, 1:szbandconst[1], p+i.+(1:szbandconst[2])))
        end
    end

    # export the accurate parts
    view(bands, 1:n-N, :)
end

function bandexpand(oldbands::AbstractMatrix{T}, lincoeffs::Vector{Vector{T}}, n::Integer, p::Integer) where {T}
    # expand the preconditioner to allow for solving n-vector
    szold = size(oldbands)
    N = length(lincoeffs) - 1
    pN = p+N

    # the region of operator we want
    rowstart = szold[1] + 1
    rowend = n - N  # minus the rows of boundary conditions

    # bands to be determined
    bands = zeros(T, rowend - rowstart + 1 + 2 * (N - 1), 2 * pN + 1) # extra 2*(N-1) rows for conversion matrices
    szbands = size(bands)
    constantterm = zeros(Integer, 0)
    for i in eachindex(lincoeffs)
        if isempty(lincoeffs[i])
            # do nothing for empty coefficients
            continue
        end

        # Note that lincoeffs[i] is not empty here
        if length(lincoeffs[i]) > 1
            # terms with variable coefficients
            bandsgenerate!(bands, lincoeffs[i], rowstart, rowend, p, i - 1, N)
        elseif i == 1
            # constant 0th order term (S_0 is multiplied)
            bands[:, p+1] .+= lincoeffs[i][1] / 2
            bands[:, p+3] .-= lincoeffs[i][1] / 2
        elseif i == 2
            # constant 1st order term
            shift = rowstart - 1
            for j = 1:szbands[1]
                bands[j, p+2] += (j + shift) * lincoeffs[i][1]
            end
        else
            # leave them along
            push!(constantterm, i - 1)
        end
    end
    # S_{N-1}...S_1(from right to left)
    for j = 1:N-1
        shift = T(rowstart + j - 1)
        scale = j ./ (shift:shift+szbands[1]-1)
        broadcast!(*, bands, bands, scale)

        # add up because two nonzero diagonals
        @views bands[1:szbands[1]-2*j, 3:szbands[2]] .-= bands[3:szbands[1]-2*(j-1), 1:szbands[2]-2]
    end

    # higher order (>=2) terms with constant coefficients
    for i in constantterm
        if !iszero(lincoeffs[i+1][1])
            bandconst = zeros(T, rowend - rowstart + 1 + 2 * (N - i), 2 * (N - i) + 1)
            if i == 0
                bandconst[:, 1] .= one(T)
            else
                bandconst[:, 1] .= 2^(i - 1) * factorial(i - 1) .* ((rowstart:rowend+2*(N-i)) .+ (i - 1))
            end
            # S_{N-1}...S_i(from right to left)
            szbandconst = size(bandconst)
            for j = i:N-1
                shift = T(rowstart + j - 1)
                scale = j ./ (shift:shift+szbandconst[1]-1)
                broadcast!(*, bandconst, bandconst, scale)

                # add up because two nonzero diagonals
                @views bandconst[1:szbandconst[1]-2*(j-i+1), 3:szbands[2]] .-= bandconst[3:szbandconst[1]-2*(j-i), 1:szbands[2]-2]
            end

            # add up
            axpy!(lincoeffs[i+1][1], bandconst, view(bands, 1:szbandconst[1], p+i.+(1:szbandconst[2])))
        end
    end

    # assign new banded parts
    newbands = Matrix{T}(undef, n - N, 2*pN + 1)
    oldinds = CartesianIndices(oldbands)
    copyto!(newbands, oldinds, oldbands, oldinds)
    copyto!(view(newbands, rowstart:rowend, :), view(bands, 1:rowend-rowstart+1, :))

    newbands
end

function bandsgenerate!(bands::AbstractMatrix{T}, lincoeff::Vector{T}, rowstart::Integer, rowend::Integer, p::Integer, lambda::Integer, N::Integer) where {T}
    # construct the preconditioner related to rows from rowstart to rowend of lambda-th order term
    # p is adaptive bandwidth for preconditioner and N is the maximum differential order and lincoeff is assumed to have no more than p + lambda + 1 elements
    # bands is assumed to be a matrix of size (rowend - rowstart + 1 + 2*(N-1)) × (2 * (p+N) + 1) and the new term is added to it. Each row and each column of bands represents a row and a diagonal of the band part of almost banded matrix, respectively.
    # NOTE: the coefficients are assumed to be properly halfed already, i.e., halfed except the first element, and of proper degree (p+lambda+1 for lambda-th order term)

    lenbands = size(bands, 1)
    plambda = p + lambda
    lenlincoeff = length(lincoeff)

    if lambda == 0
        # the lowest term
        bandnow = Matrix{T}(undef, lenbands + 2, 2 * p + 1)
        # M_0
        # Toeplitz part
        bandnow[:, p+1] .= lincoeff[1]
        for k = 2:lenlincoeff
            bandnow[:, p+k] .= lincoeff[k]
            bandnow[:, p+2-k] .= lincoeff[k]
        end
        for k = lenlincoeff+1:p+1
            bandnow[:, p+k] .= 0
            bandnow[:, p+2-k] .= 0
        end
        # Hankel part
        if rowstart < plambda + 1
            for j = max(2, rowstart):min(plambda + 1, rowend)
                bandnow[j, (1:lenlincoeff-j+1).+(plambda+1-j)] .+= view(lincoeff, j:lenlincoeff)
            end
        end
        # zero out redundant elements (outside the upperleft corner of the matrix)
        for i = rowstart:plambda
            bandnow[i, 1:plambda+1-i] .= zero(T)
        end

        # S_0 * M_0
        ldiv!(2, bandnow)
        if rowstart == 1
            rmul!(view(bandnow, 1, :), 2)
        end
        axpy!(true, view(bandnow, 1:lenbands, 1:2*p+1), view(bands, 1:lenbands, 1:2*p+1))
        axpy!(-one(T), view(bandnow, 3:lenbands+2, 1:2*p+1), view(bands, 1:lenbands, 3:2*p+3))
    else
        # higher terms
        bandnow = zero(bands)
        # M_1
        # Toeplitz part
        bandnow[:, plambda+1] .= lincoeff[1]
        for k = 2:lenlincoeff
            bandnow[:, plambda+k] .= lincoeff[k]
            bandnow[:, plambda+2-k] .= lincoeff[k]
        end
        for k = lenlincoeff+1:plambda+1
            bandnow[:, plambda+k] .= 0
            bandnow[:, plambda+2-k] .= 0
        end
        # Hankel part
        if rowstart < plambda
            for j = rowstart:min(plambda - 1, rowend)
                bandnow[j, (1:lenlincoeff-j-1).+(plambda+1-j)] .-= view(lincoeff, j+2:lenlincoeff)
            end
        end
        # zero out redundant elements (outside the upperleft corner of the matrix)
        for i = rowstart:plambda
            bandnow[i, 1:plambda+1-i] .= zero(T)
        end

        # M_1 S_1^{-1}...S_{λ-1}^{-1}
        for i = 1:lambda-1
            shift = i + rowstart - 1
            scale = Vector{T}((shift-plambda:shift+plambda+lenbands-1) ./ i)
            for l = 1:lenbands
                temp = view(bandnow, l, 1:2*plambda+1)
                broadcast!(*, temp, temp, view(scale, l:l+2*plambda))
            end
            # add up columnwise (cannot do at the same time because of the structure of S^{-1})
            for m = 1:size(bandnow, 2)-2
                axpy!(true, view(bandnow, 1:lenbands, m), view(bandnow, 1:lenbands, m + 2))
            end
        end
        # D_{λ}
        shift = lambda + rowstart - 1
        scale = Vector{T}(shift-plambda:shift+plambda+2*(N-lambda)+lenbands-1)
        for l = 1:lenbands
            temp = view(bandnow, l, :)
            broadcast!(*, temp, temp, view(scale, l:l+2*(p+N)))
        end
        lmul!(2^(lambda - 1) * factorial(lambda - 1), bandnow) # constant

        # add up
        axpy!(true, bandnow, bands)
    end

    bands
end

function almostbanded_init(bands::AbstractMatrix{T}, bc::AbstractMatrix{T}, p::Integer) where T
    # construct the AlmostBandedMatrix for Jacobian system
    n, N = size(bc)  # determine the parameter
    pN = p+N
    banded = BandedMatrix(Zeros(T, n, n), (pN, 2*pN))  # reserved for in-place qr

    # assign boundary conditions for first N rows
    for i = 1:N
        banded[i, 1:2*pN+i] .= view(bc, 1:2*pN+i, i)
    end

    # assign operator parts
    opbanded = view(banded, N+1:n, :)
    for i = -p:-1
        opbanded[band(i)] .= view(bands, 1-i:n-N, i+p+1)
    end
    for i = 0:N
        opbanded[band(i)] .= view(bands, 1:n-N, i+p+1)
    end
    for i = N+1:p+2*N
        opbanded[band(i)] .= view(bands, 1:n-i, i+p+1)
    end

    AlmostBandedMatrix(banded, ApplyMatrix(*, Matrix{T}(I, n, N), transpose(bc)))
end