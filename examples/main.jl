## load the INGU framework for solving nonlinear ODEs
include("..\\src\\INGU.jl")

## load the examples
include("Examples.jl")

## all the problems
@show Jvec


## testing the speed using BenchmarkTools (This may take long time but it is worthwhile to see the power of INGU!)
# NOTE: @belapsed takes the minimum execution time among repeated executions.
using BenchmarkTools
begin
    TRD_timing = zeros(Float64, length(Jvec))
    LSB_timing = zeros(Float64, length(Jvec))
    TRC_timing = zeros(Float64, length(Jvec))
    TRD_uvec = Vector{Vector{Float64}}(undef, length(Jvec))
    LSB_uvec = Vector{Vector{Float64}}(undef, length(Jvec))
    TRC_uvec = Vector{Vector{Float64}}(undef, length(Jvec))
    TRD_res = similar(TRD_timing)
    LSB_res = similar(LSB_timing)
    TRC_res = similar(LSB_timing)
    for i in eachindex(Jvec)
        TRD_uvec[i], TRD_res[i] = TRD(Jvec[i])
        LSB_uvec[i], LSB_res[i] = LSB(Jvec[i])
        TRC_uvec[i], TRC_res[i] = TRC(Jvec[i])
        TRD_timing[i] = @belapsed TRD($(Jvec[i]))
        LSB_timing[i] = @belapsed LSB($(Jvec[i]))
        TRC_timing[i] = @belapsed TRC($(Jvec[i]))

        # show the results
        @printf "The execution time for %s problem using TRD method is %.2es, the final residual is %.2e and the length of the solution is %i.\n\n" string(Jvec[i]) TRD_timing[i] TRD_res[i] length(TRD_uvec[i])
        @printf "The execution time for %s problem using LSB method is %.2es, the final residual is %.2e and the length of the solution is %i.\n\n" string(Jvec[i]) LSB_timing[i] LSB_res[i] length(LSB_uvec[i])
        @printf "The execution time for %s problem using TRC method is %.2es, the final residual is %.2e and the length of the solution is %i.\n\n" string(Jvec[i]) TRC_timing[i] TRC_res[i] length(TRC_uvec[i])
    end
end



# visualizing solutions
using Plots
for k in eachindex(Jvec)
    unow, _ = TRC(Jvec[k])
    Base.display(plot(chebpts(length(unow)), coeffs2vals(unow), xlabel = "x", ylabel = "u", label = string("The solution of ", string(Jvec[k]), " equation")))
end


# an integral result for Gulf problem I = ∫_0^∞ [(u'')^2 - 3*lambda*u*u'*u''] dx = 1/2
begin
    uGulf, _ = TRC(Gulf)
    lambda = -0.1
    f1 = fastmult!(fastdive(uGulf, 2), fastdive(uGulf, 2))
    f2 = fastmult!(fastmult!(fastdive(uGulf, 1), fastdive(uGulf, 2)), uGulf)
    coeffsadd_chop!(-3*(35/2)*lambda, f2, f1)
    # the integral
    @assert fastquad(f1) * (2/35)^3 ≈ 1/2 "Identity fails."
end

# path-following
uvec, lamvec, utangentvec, lamtangentvec, uprevec, lamprevec = pathfollowing(Sawtooth, Sawtoothdlam, 5e-2, 5e-5)