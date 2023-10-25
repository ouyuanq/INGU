# the examples for nonlinear solver
# Note that multiple dispatch is used.

## Painleve equation
# u'' - 25 * u^2 + 125 * (x + 1) = 0, x ∈ [-1, 1]
# u(-1) = 0, u(1) = √10
function Painleve(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ T.([0.; sqrt(10)])
end

function Painleve(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    n0 = lmul!(-25, fastsquare!(copy(u)))  # the nonlinear term
    n0[1:2] .+= T(125.)
    fastconv!(n0, 0, 2)

    coeffsadd_chop!(true, n2, n0)

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[2] = reduce(+, u)
    n0[2] -= sqrt(10)

    n0
end

# Jacobian: δ'' - 50 * u * δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Painleve(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(-50, u)
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Painleve(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Painleve(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Painleve(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Carrier equation
# 0.01*u'' + 2*(1-x^2)*u + u^2 - 1 = 0, x ∈ [-1, 1]
# u(-1) = 0, u(1) = 0
function Carrier(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ zeros(T, 2)
end

function Carrier(u::AbstractVector{T}; epsilon=T(0.01)) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    
    c0 = coeffsadd!(true, [1; 0; -1], copy(u))
    n0 = fastmult!(c0, u)  # the nonlinear term
    n0[1] -= one(T)
    
    coeffsadd_chop!(epsilon, n2, fastconv!(n0, 0, 2))  # to C^{(2)} series and add up

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[2] = reduce(+, u)

    n0
end

# Jacobian: 0.01*δ'' + 2*(1-x^2)*δ + 2*u*δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Carrier(u::AbstractVector{T}, ::AbstractChar; epsilon=T(0.01)) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(2, u)
    coeffsadd!(true, [1; 0; -1], JacCoeffs[1])
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = lmul!(epsilon, ones(T, 1))

    standardChop!.(JacCoeffs)
end

function Carrier(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Carrier(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Carrier(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## InteriorLayer equation
# 0.04*u'' + 2*u*u' + u = 0, x ∈ [-1, 1]
# u(-1) = -7/6, u(1) = 3/2
function InteriorLayer(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ T.([-7/6; 3/2])
end

function InteriorLayer(u::AbstractVector{T}; epsilon=T(0.04)) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term

    c0 = lmul!(T(2), fastdiff(u, 1))  # C^{(1)} series
    c0[1] += one(T)
    n0 = fastmult1!(c0, u) # the nonlinear term

    coeffsadd_chop!(epsilon, n2, fastconv!(n0, 1, 2))

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] += T(7/6)
    n0[2] = reduce(+, u)
    n0[2] -= T(3/2)

    n0
end

# Jacobian: 0.04*δ'' + 2*u*δ' + 2*u'*δ + δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function InteriorLayer(u::AbstractVector{T}, ::AbstractChar; epsilon=T(0.04)) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(T(2), fastdive(u, 1))
    JacCoeffs[1][1] += 1
    JacCoeffs[2] = lmul!(2, u)
    JacCoeffs[3] = lmul!(epsilon, ones(T, 1))

    standardChop!.(JacCoeffs)
end

function InteriorLayer(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function InteriorLayer(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function InteriorLayer(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## BoundaryLayer equation
# 0.04 * u'' + 2*u*u' - (0.5*x + 0.5)*u = 0, x ∈ [-1, 1]
# u(-1) = -7/6, u'(1) = 3/4
function BoundaryLayer(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightN!(A[2, :], 1)
    end

    A \ T.([-7/6; 3/4])
end

function BoundaryLayer(u::AbstractVector{T}; epsilon=T(0.04)) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term

    c0 = lmul!(T(2), fastdive(u, 1))
    coeffsadd!(true, ldiv!(-2, ones(T, 2)), c0)  # Chebyshev T series here
    n0 = fastmult!(c0, u) # the nonlinear term

    coeffsadd_chop!(epsilon, n2, fastconv!(n0, 0, 2))

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] += T(7/6)
    # Neumann boundary conditions
    uN = similar(u)
    for i in eachindex(uN)
        uN[i] = (i-1)^2 * u[i]
    end
    n0[2] = reduce(+, uN) - T(3/4)

    n0
end

# Jacobian: 0.04*δ'' + 2*u*δ' + 2*u'*δ - (0.5*x + 0.5)*δ = 0, δ(-1) = δ'(1) = 0, δ ∈[-1, 1]
function BoundaryLayer(u::AbstractVector{T}, ::AbstractChar; epsilon=T(0.04)) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(T(2), fastdive(u, 1))
    coeffsadd!(true, ldiv!(-2, ones(T, 2)), JacCoeffs[1])  # Chebyshev T series here
    JacCoeffs[2] = lmul!(2, u)
    JacCoeffs[3] = lmul!(epsilon, ones(T, 1))

    standardChop!.(JacCoeffs)
end

function BoundaryLayer(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    # Neumann boundary conditions
    dN = similar(delta)
    for i in eachindex(dN)
        dN[i] = (i-1)^2 * delta[i]
    end
    g[2] = reduce(+, dN)

    g
end

function BoundaryLayer(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + (i-1)^2*FC[2]
    end

    g
end

function BoundaryLayer(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightN!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Birkisson1 equation
# u'' - pi/4*cos(pi/4*(x+1))*u' + (pi/4)^2*u*log(u) = 0, x ∈ [-1, 1]
# u(-1) = 1, u(1) = ℯ
function Birkisson1(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ T.([1.; exp(1)])
end

function Birkisson1(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term

    c1 = lmul!(T(-pi/4), mycos(lmul!(pi/4, ones(T, 2))))
    n1 = fastmult1!(fastdiff(u, 1), c1)  # the 1st order term

    c0 = mylog(u)
    n0 = fastmult!(c0, u)  # the nonlinear term
    lmul!(T((pi/4)^2), fastconv!(n0, 0, 1))

    coeffsadd!(true, n1, n0)  # C^{(1)} series

    coeffsadd_chop!(true, n2, fastconv!(n0, 1, 2))   # to C^{(2)} series and add up

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] -= one(T)
    n0[2] = reduce(+, u)
    n0[2] -= T(exp(1))

    n0
end

# Jacobian: δ'' - pi/4*cos(pi/4*(x+1))*δ' + (pi/4)^2*(log(u) + 1)*δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Birkisson1(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = mylog(u)
    JacCoeffs[1][1] += 1
    lmul!(T((-pi/4)^2), JacCoeffs[1])
    JacCoeffs[2] = lmul!(T(-pi/4), mycos(lmul!(pi/4, ones(T, 2))))
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Birkisson1(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Birkisson1(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Birkisson1(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Birkisson2 equation
# u'' - 1.25*u' + 25/16 * (exp(2.5*x +2.5)*u + u^2 - sin(exp(1.25*x + 1.25))^2) = 0, x ∈ [-1, 1]
# u(-1) = sin(1), u(1) = sin(exp(2.5))
function Birkisson2(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ T.([sin(1); sin(exp(2.5))])
end

function Birkisson2(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term

    n1 = fastdiff(u, 1)  # the 1st order term

    # constant coefficients declared at the beginning of this file
    c0 = fastsquare!(mysin(myexp(lmul!(1.25, ones(T, 2)))))
    c1 = coeffsadd!(true, u, myexp(lmul!(2.5, ones(T, 2))))
    n0 = fastmult!(c1, u)
    coeffsadd!(-one(T), c0, n0)
    lmul!(T(25/16), n0)  # the 0th order term

    coeffsadd!(T(-1.25), n1, fastconv!(n0, 0, 1)) # C^{(1)} series
    coeffsadd_chop!(true, n2, fastconv!(n0, 1, 2))  # C^{(2)} series

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] -= T(sin(1))
    n0[2] = reduce(+, u)
    n0[2] -= T(sin(exp(2.5)))

    n0
end

# Jacobian: δ'' - 1.25*δ' + 25/16 * (exp(2.5*x +2.5)*δ + 2*u*δ) = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Birkisson2(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(T(25/16), coeffsadd!(true, myexp(lmul!(2.5, ones(T, 2))), lmul!(2, u)))
    JacCoeffs[2] = [T(-1.25)]
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Birkisson2(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Birkisson2(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Birkisson2(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Birkisson3 equation
# u'' + 18*(u - u^3) = 0, x ∈ [-1, 1]
# u(-1) = tanh(-3), u(0) = tanh(0)
function Birkisson3(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        # rightD!(A[2, :], 1)
        middle!(A[2, :], 1)
    end

    # A \ T.([-tanh(3); tanh(3)])
    A \ T.([-tanh(3); tanh(0)])
end

function Birkisson3(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    
    c0 = fastsquare!(copy(u))
    c0[1] -= one(T)
    n0 = fastmult!(c0, u)  # the nonlinear term
    lmul!(T(-18), n0)
    
    coeffsadd_chop!(true, n2, fastconv!(n0, 0, 2))  # to C^{(2)} series and add up

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] += tanh(3)
    # n0[2] = reduce(+, u)
    # n0[2] -= tanh(3)
    n0[2] = foldr(-, view(u, 1:2:length(u)))
    n0[2] -= tanh(0)

    n0
end

# Jacobian: δ'' + 18*(1 - 3*u^2)*δ = 0, δ(-1) = δ(0) = 0, δ ∈[-1, 1]
function Birkisson3(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(-54, fastsquare!(u))
    JacCoeffs[1][1] += T(18)
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Birkisson3(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    # g[2] = reduce(+, delta)
    g[2] = foldr(-, view(delta, 1:2:length(delta)))

    g
end

function Birkisson3(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + (mod(i, 2) * (2 - mod(i, 4)))*FC[2]
    end

    g
end

function Birkisson3(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        # rightD!(bcnew[:, 2], szbc[1] + 1)
        middle!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Fisher equation
# u'' + 16*(u - u^2) = 0, x ∈ [-1, 1]
# u(-1) = 1, u(1) = 0
function Fisher(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ [1; 0]
end

function Fisher(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    
    n0 = fastsquare!(copy(u))  # the nonlinear term
    coeffsadd!(-one(T), u, n0)
    lmul!(T(-16), n0)
    
    coeffsadd_chop!(true, n2, fastconv!(n0, 0, 2))  # to C^{(2)} series and add up

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] -= one(T)
    n0[2] = reduce(+, u)

    n0
end

# Jacobian: δ'' + (16 - 32*u)*δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Fisher(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(-32, u)
    JacCoeffs[1][1] += T(16)
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Fisher(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Fisher(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Fisher(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Blasius equation
# u''' + (L/4)*u*u'' = 0, x ∈ [-1, 1]
# u(-1) = 0, u'(-1) = 0, u'(1) = L/2
function Blasius(::Type{T}; L = 10) where T
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 3, 3)
    @views begin
        leftD!(A[1, :], 1)
        leftN!(A[2, :], 1)
        rightN!(A[3, :], 1)
    end

    A \ [0; 0; L/2]
end

function Blasius(u::AbstractVector{T}; L = 10) where T
    # evaluate the nonlinear operator
    n3 = fastdiff(u, 3)  # the 3rd order term

    prod = fastmult1!(fastdive2U(u, 2), u)  # the product term
    lmul!(L/4, prod)

    # Note that product is a C^{(1)} series
    coeffsadd_chop!(true, n3, fastconv!(prod, 1, 3))

    prepend!(prod, zeros(T, 3))
    # Dirichlet boundary conditions
    prod[1] = foldr(-, u)
    # Neumann boundary conditions
    uN = similar(u)
    for i in eachindex(uN)
        uN[i] = (i-1)^2 * u[i]
    end
    prod[2] = -foldr(-, uN)
    prod[3] = reduce(+, uN)
    prod[3] -= L/2

    prod
end

# Jacobian: δ''' + 2.5*u*δ'' + 2.5*u''*δ = 0, δ(-1) = δ'(-1) = δ'(1) = 0, δ ∈[-1, 1]
function Blasius(u::AbstractVector{T}, ::AbstractChar; L = 10) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 4)
    JacCoeffs[1] = lmul!(L/4, fastdive(u, 2))
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = lmul!(L/4, u)
    JacCoeffs[4] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Blasius(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 4:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:3] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    # Neumann boundary conditions
    dN = similar(delta)
    for i in eachindex(dN)
        dN[i] = (i-1)^2 * delta[i]
    end
    g[2] = -foldr(-, dN)
    g[3] = reduce(+, dN)

    g
end

function Blasius(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        j = (i-1)^2
        g[i] += (-1)^(i - 1)*FC[1] + (-1)^i*j*FC[2] + j*FC[3]
    end

    g
end

function Blasius(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        leftN!(bcnew[:, 2], szbc[1] + 1)
        rightN!(bcnew[:, 3], szbc[1] + 1)
    end

    bcnew
end


## Falkner equation
# u''' + (L/4)*u*u'' - (L/3)*(u')^2 + (L^3/12) = 0, x ∈ [-1, 1]
# u(-1) = 0, u'(-1) = 0, u'(1) = L/2
function Falkner(::Type{T}; L = 10) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 3, 3)
    @views begin
        leftD!(A[1, :], 1)
        leftN!(A[2, :], 1)
        rightN!(A[3, :], 1)
    end

    A \ [0; 0; L/2]
end

function Falkner(u::AbstractVector{T}; L = 10) where T
    # evaluate the nonlinear operator
    n3 = fastdiff(u, 3)  # the 3rd order term

    prod = fastmult1!(fastdive2U(u, 2), u)  # the product term
    lmul!(L/4, prod)
    prod2 = fastmult1!(fastdiff(u, 1), fastdive(u, 1))

    # Note that product is a C^{(1)} series
    coeffsadd!(-L/3, prod2, prod)
    coeffsadd_chop!(true, n3, fastconv!(prod, 1, 3))
    prod[1] += L^3/12

    prepend!(prod, zeros(T, 3))
    # Dirichlet boundary conditions
    prod[1] = foldr(-, u)
    # Neumann boundary conditions
    uN = similar(u)
    for i in eachindex(uN)
        uN[i] = (i-1)^2 * u[i]
    end
    prod[2] = -foldr(-, uN)
    prod[3] = reduce(+, uN)
    prod[3] -= L/2

    prod
end

# Jacobian: δ''' + 2.5*u*δ'' + 2.5*u''*δ - 20/3*u'*δ' = 0, δ(-1) = δ'(-1) = δ'(1) = 0, δ ∈[-1, 1]
function Falkner(u::AbstractVector{T}, ::AbstractChar; L = 10) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 4)
    JacCoeffs[1] = lmul!(L/4, fastdive(u, 2))
    JacCoeffs[2] = lmul!(-2*L/3, fastdive(u, 1))
    JacCoeffs[3] = lmul!(L/4, u)
    JacCoeffs[4] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Falkner(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 4:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:3] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    # Neumann boundary conditions
    dN = similar(delta)
    for i in eachindex(dN)
        dN[i] = (i-1)^2 * delta[i]
    end
    g[2] = -foldr(-, dN)
    g[3] = reduce(+, dN)

    g
end

function Falkner(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        j = (i-1)^2
        g[i] += (-1)^(i - 1)*FC[1] + (-1)^i*j*FC[2] + j*FC[3]
    end

    g
end

function Falkner(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        leftN!(bcnew[:, 2], szbc[1] + 1)
        rightN!(bcnew[:, 3], szbc[1] + 1)
    end

    bcnew
end


## Gulf equation
# 4/7*u''' + (u')^2 - u*u'' - 10*(35/2)^2*(u-1) = 0, x ∈ [-1, 1]
# u(-1) = 0, u''(-1) = 0, u(1) = 1
function Gulf(::Type{T}) where T
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 3, 3)
    @views begin
        leftD!(A[1, :], 1)
        leftDer2!(A[2, :], 1)
        rightD!(A[3, :], 1)
    end

    A \ [0; 0; 1]
end

function Gulf(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n3 = fastdiff(u, 3)  # the 3rd order term

    prod = fastmult1!(fastdiff(u, 1), fastdive(u, 1))  # the product term in C^{(1)} basis
    c1 = fastdive2U(u, 2)
    c1[1] += T(10*(35/2)^2)
    prod2 = fastmult1!(c1, u)  # the product term in C^{(1)} basis
    coeffsadd!(-one(T), prod2, prod)
    prod[1] += T(10*(35/2)^2)

    # Note that product is a C^{(1)} series
    coeffsadd_chop!(T(4/7), n3, fastconv!(prod, 1, 3))

    prepend!(prod, zeros(T, 3))
    # Dirichlet boundary conditions
    prod[1] = foldr(-, u)
    prod[3] = reduce(+, u)
    prod[3] -= one(T)
    # 2nd order derivative boundary conditions
    prod[2] = leftDer2(u)

    prod
end

# Jacobian: 4/7*δ''' - u*δ'' + 2*u'*δ' - u''*δ - 10*(35/2)^2*δ = 0, δ(-1) = δ''(-1) = δ(1) = 0, δ ∈[-1, 1]
function Gulf(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 4)
    JacCoeffs[1] = lmul!(-one(T), fastdive(u, 2))
    JacCoeffs[1][1] -= T(10*(35/2)^2)
    JacCoeffs[2] = lmul!(T(2), fastdive(u, 1))
    JacCoeffs[3] = lmul!(-one(T), u)
    JacCoeffs[4] = lmul!(T(4/7), ones(T, 1))

    standardChop!.(JacCoeffs)
end

function Gulf(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 4:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:3] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[3] = reduce(+, delta)
    # 2nd order derivative boundary conditions
    g[2] = leftDer2(delta)

    g
end

function Gulf(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        a, b = (-1)^(i - 1), (i - 1)^2
        g[i] += a*FC[1] + a*b*(b-1)/3*FC[2] + FC[3]
    end

    g
end

function Gulf(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        leftDer2!(bcnew[:, 2], szbc[1] + 1)
        rightD!(bcnew[:, 3], szbc[1] + 1)
    end

    bcnew
end


## Fourth equation
# 2*u'''' - u'*u'' + u*u''' = 0, x ∈ [-1, 1]
# u(-1) = 0, u'(-1) = 0, u(1) = 1, u'(1) = -5/2
function Fourth(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 4, 4)
    @views begin
        leftD!(A[1, :], 1)
        leftN!(A[2, :], 1)
        rightD!(A[3, :], 1)
        rightN!(A[4, :], 1)
    end

    A \ T.([0; 0; 1.; -5/2])
end

function Fourth(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n4 = fastdiff(u, 4)  # the 4th order term

    prod = fastmult1!(fastdive2U(u, 3), u)   # the product term
    prod2 = fastmult1!(fastdiff(u, 1), fastdive(u, 2))

    # Note that product is a C^{(1)} series
    coeffsadd!(-one(T), prod2, prod)
    coeffsadd_chop!(T(2), n4, fastconv!(prod, 1, 4))

    prepend!(prod, zeros(T, 4))
    # Dirichlet boundary conditions
    prod[1] = foldr(-, u)
    prod[3] = reduce(+, u)
    prod[3] -= one(T)
    # Neumann boundary conditions
    uN = similar(u)
    for i in eachindex(uN)
        uN[i] = (i-1)^2 * u[i]
    end
    prod[2] = -foldr(-, uN)
    prod[4] = reduce(+, uN)
    prod[4] += T(5/2)

    prod
end

# Jacobian: 2*δ'''' - u'*δ'' - u''*δ' + u*δ''' + u'''*δ = 0, δ(-1) = δ'(-1) = δ(1) = δ'(1) = 0, δ ∈[-1, 1]
function Fourth(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 5)
    JacCoeffs[1] = fastdive(u, 3)
    JacCoeffs[2] = lmul!(-one(T), fastdive(u, 2))
    JacCoeffs[3] = lmul!(-one(T), fastdive(u, 1))
    JacCoeffs[4] = u
    JacCoeffs[5] = lmul!(T(2), ones(T, 1))

    standardChop!.(JacCoeffs)
end

function Fourth(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 5:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:4] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[3] = reduce(+, delta)
    # Neumann boundary conditions
    dN = similar(delta)
    for i in eachindex(dN)
        dN[i] = (i-1)^2 * delta[i]
    end
    g[2] = -foldr(-, dN)
    g[4] = reduce(+, dN)

    g
end

function Fourth(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        b = (i - 1)^2
        g[i] += (-1)^(i - 1)*FC[1] + (-1)^i*b*FC[2] + FC[3] + b*FC[4]
    end

    g
end

function Fourth(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        leftN!(bcnew[:, 2], szbc[1] + 1)
        rightD!(bcnew[:, 3], szbc[1] + 1)
        rightN!(bcnew[:, 4], szbc[1] + 1)
    end

    bcnew
end


## Bratu equation
# u'' + 0.875*exp(u) = 0, x ∈ [-1, 1]
# u(-1) = 0, u(1) = 0
function Bratu(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ zeros(T, 2)
end

function Bratu(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    n0 = lmul!(T(0.875), myexp(u))  # the 0th order term

    coeffsadd_chop!(true, n2, fastconv!(n0, 0, 2))

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[2] = reduce(+, u)

    n0
end

# Jacobian: δ'' + 0.875*exp(u)*δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Bratu(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(T(0.875), myexp(u))
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Bratu(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Bratu(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Bratu(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## Pendulum equation
# u'' + 25*sin(u) = 0, x ∈ [-1, 1]
# u(-1) = 2, u(1) = 2
function Pendulum(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ [2; 2]

    # coeffs(x -> 2*cos(pi*(x+1)), 20, T)
end

function Pendulum(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    n0 = lmul!(T(25), mysin(u))  # the nonlinear term

    coeffsadd_chop!(true, n2, fastconv!(n0, 0, 2))

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)-2
    n0[2] = reduce(+, u)-2

    n0
end

# Jacobian: δ'' + 25*cos(u)*δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Pendulum(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(T(25), mycos(u))
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = ones(T, 1)

    standardChop!.(JacCoeffs)
end

function Pendulum(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Pendulum(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Pendulum(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end



## Sawtooth equation
# 0.05*u'' + (u')^2 - 1 = 0, x ∈ [-1, 1]
# u(-1) = 0.8, u(1) = 1.2
function Sawtooth(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ T.([.8; 1.2])
end

function Sawtooth(u::AbstractVector{T}; epsilon=T(0.05)) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    n0 = fastmult1!(fastdiff(u, 1), fastdive(u, 1))  # the nonlinear term
    n0[1] -= one(T)

    coeffsadd_chop!(epsilon, n2, fastconv!(n0, 1, 2))

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u) - T(0.8)
    n0[2] = reduce(+, u) - T(1.2)

    n0
end

# Jacobian: 0.05*δ'' + 2*u'*δ' = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function Sawtooth(u::AbstractVector{T}, ::AbstractChar; epsilon=T(0.05)) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = zeros(T, 0)
    JacCoeffs[2] = lmul!(T(2), fastinve!(fastdiff!(u, 1), 1))
    JacCoeffs[3] = lmul!(epsilon, ones(T, 1))

    standardChop!.(JacCoeffs)
end

function Sawtooth(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function Sawtooth(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function Sawtooth(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## LaneEmden equation
# (x+1)*u'' + 2*u' + (25*x+25)*u^5 = 0, x ∈ [-1, 1]
# u(-1) = 1, u'(-1) = 0
function LaneEmden(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        leftN!(A[2, :], 1)
    end

    standardChop!(A \ [1; 0])
end

function LaneEmden(u::AbstractVector{T}) where T
    # evaluate the nonlinear operator
    n2 = fastmult1!(fastdive2U(u, 2), ones(T, 2))  # the 2nd order term
    n1 = fastdiff(u, 1)
    coeffsadd!(T(2), n1, n2)

    n0 = fastsquare!(fastsquare!(copy(u)))
    d0 = fastmult!(copy(u), lmul!(25, ones(T, 2)))
    fastmult!(n0, d0)  # the nonlinear term

    coeffsadd!(true, n2, fastconv!(n0, 0, 1))

    fastconv!(n0, 1, 2)   # to C^{(2)} series
    deleteat!(n0, standardChop(n0)+1:length(n0))

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] -= one(T)
    # Neumann boundary conditions
    uN = similar(u)
    for i in eachindex(uN)
        uN[i] = (i-1)^2 * u[i]
    end
    n0[2] = -foldr(-, uN)

    n0
end

# Jacobian: (x+1)*δ'' + 2*δ' + (125*x+125)*u^4*δ = 0, δ(-1) = δ'(-1) = 0, δ ∈[-1, 1]
function LaneEmden(u::AbstractVector{T}, ::AbstractChar) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    n0 = fastsquare!(fastsquare!(u))
    JacCoeffs[1] = fastmult!(n0, lmul!(125, ones(T, 2)))
    JacCoeffs[2] = lmul!(T(2), ones(T, 1))
    JacCoeffs[3] = ones(T, 2)

    standardChop!.(JacCoeffs)
end

function LaneEmden(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    # Neumann boundary conditions
    dN = similar(delta)
    for i in eachindex(dN)
        dN[i] = (i-1)^2 * delta[i]
    end
    g[2] = -foldr(-, dN)

    g
end

function LaneEmden(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + (-1)^i*(i - 1)^2*FC[2]
    end

    g
end

function LaneEmden(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        leftN!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


## AllenCahn equation
# 0.08*u'' + u - u^3 - sin(5*x + 5) = 0, x ∈ [-1, 1]
# u(-1) = 1, u(1) = -1
function AllenCahn(::Type{T}) where T 
    # get the initial iteration which is a low degree polynomial satisfies boundary condition
    A = Matrix{T}(undef, 2, 2)
    @views begin
        leftD!(A[1, :], 1)
        rightD!(A[2, :], 1)
    end

    A \ [1; -1]
end

function AllenCahn(u::AbstractVector{T}; epsilon = T(0.08)) where T
    # evaluate the nonlinear operator
    n2 = fastdiff(u, 2)  # the 2nd order term
    
    c0 = lmul!(-one(T), fastsquare!(copy(u)))
    c0[1] += one(T)
    n0 = fastmult!(c0, u)  # the nonlinear term

    coeffsadd!(-one(T), mysin(lmul!(5, ones(T, 2))), n0)
    
    coeffsadd_chop!(epsilon, n2, fastconv!(n0, 0, 2))  # to C^{(2)} series and add up

    prepend!(n0, zeros(T, 2))
    # Dirichlet boundary conditions
    n0[1] = foldr(-, u)
    n0[1] -= one(T)
    n0[2] = reduce(+, u)
    n0[2] += one(T)

    n0
end

# Jacobian: 0.08*δ'' + δ - 3*u^2*δ = 0, δ(-1) = δ(1) = 0, δ ∈[-1, 1]
function AllenCahn(u::AbstractVector{T}, ::AbstractChar; epsilon = T(0.08)) where T
    # get the coefficients for Jacobian operator at u
    JacCoeffs = Vector{Vector{T}}(undef, 3)
    JacCoeffs[1] = lmul!(-3, fastsquare!(u))
    JacCoeffs[1][1] += one(T)
    JacCoeffs[2] = zeros(T, 0)
    JacCoeffs[3] = lmul!(epsilon, ones(T, 1))

    standardChop!.(JacCoeffs)
end

function AllenCahn(g::AbstractVector{T}, JacCoeffs::Vector{Vector{T}}, delta::AbstractVector{T}, Plan, Plani) where T
    # evaluate the boundary conditions and Jacobian operator
    JacEval!(view(g, 3:length(g)), JacCoeffs, delta, Plan, Plani)

    g[1:2] .= zero(T)
    # Dirichlet boundary conditions
    g[1] = foldr(-, delta)
    g[2] = reduce(+, delta)

    g
end

function AllenCahn(::AbstractChar, JacCoeffs::Vector{Vector{T}}, FC::AbstractVector, Plan::FFTW.cFFTWPlan, Plani::AbstractFFTs.ScaledPlan) where T
    # apply the transpose of Jacobian operator to FC
    g = JacEval_trans(JacCoeffs, view(FC, length(JacCoeffs):min(length(FC), cld(length(Plan), 2)+length(JacCoeffs)-1)), Plan, Plani)

    # boundary conditions
    @inbounds for i in eachindex(g)
        g[i] += (-1)^(i - 1)*FC[1] + FC[2]
    end

    g
end

function AllenCahn(::AbstractChar, bc::AbstractMatrix{T}, n::Integer) where T
    # prolong the boundary conditions
    szbc = size(bc)
    bcnew = Matrix{T}(undef, n, szbc[2])
    bcinds = CartesianIndices(bc)
    copyto!(bcnew, bcinds, bc, bcinds)
    @views begin
        leftD!(bcnew[:, 1], szbc[1] + 1)
        rightD!(bcnew[:, 2], szbc[1] + 1)
    end

    bcnew
end


# all the problems
Jvec = Vector{Function}(undef, 0)
push!(Jvec, Blasius)
push!(Jvec, Falkner)
push!(Jvec, Fisher)
push!(Jvec, Fourth)
push!(Jvec, Bratu)
push!(Jvec, LaneEmden)
push!(Jvec, Gulf)
push!(Jvec, InteriorLayer)
push!(Jvec, BoundaryLayer)
push!(Jvec, Sawtooth)
push!(Jvec, AllenCahn)
push!(Jvec, Pendulum)
push!(Jvec, Carrier)
push!(Jvec, Painleve)
push!(Jvec, Birkisson1)
push!(Jvec, Birkisson2)
push!(Jvec, Birkisson3)


# used in path-following method
function Sawtoothdlam(u::AbstractVector{T}) where T
    # lambda*u'' + (u')^2 - 1 = 0, x ∈ [-1, 1]
    # u(-1) = 0.8, u(1) = 1.2

    # dN/dlam
    # u'' = 0, x ∈ [-1, 1]
    # u(-1) = 0, u(1) = 0

    fastdiff!(u, 2)
    prepend!(u, zeros(T, 2))  # boundary conditions
    u
end