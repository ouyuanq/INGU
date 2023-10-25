## get the coefficients for boundary conditions

# left Dirichlet condition
function leftD!(x::AbstractVector{T}, xstart::Integer) where T
    xend = length(x)
    @inbounds for i = xstart:xend
        x[i] = 2 * mod(i, 2) - 1
    end
end

# right Dirichlet condition
function rightD!(x::AbstractVector{T}, xstart::Integer) where T
    xend = length(x)
    x[xstart:xend] .= one(T)
end

# left Neumann condition
function leftN!(x::AbstractVector{T}, xstart::Integer) where T
    xend = length(x)
    @inbounds for i = xstart:xend
        x[i] = (-1)^i * (i-1)^2
    end
end

# right Neumann condition
function rightN!(x::AbstractVector{T}, xstart::Integer) where T
    xend = length(x)
    @inbounds for i = xstart:xend
        x[i] = (i-1)^2
    end
end

# left 2nd order derivative boundary condition
function leftDer2!(x::AbstractVector{T}, xstart::Integer) where T
    xend = length(x)
    @inbounds for i = xstart:xend
        temp = (i-1)^2
        x[i] = (-1)^(i-1) * temp * (temp - 1) / T(3)
    end
end

function leftDer2(x::AbstractVector{T}) where T
    # evaluate 2nd order derivative of function x at left end point
    xval = zero(T)
    for i in eachindex(x)
        temp = (i-1)^2
        xval += ((-1)^(i-1) * temp * (temp - 1) / T(3)) * x[i]
    end

    xval
end

# middle point condition
function middle!(x::AbstractVector{T}, xstart::Integer) where T
    xend = length(x)
    @inbounds for i = xstart:xend
        x[i] = mod(i, 2) * (2 - mod(i, 4))
    end
end