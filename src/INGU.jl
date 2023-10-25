# solve nonlinear ODEs using inexact Newton-GMRES-ultraspherical (INGU) framework
using LinearAlgebra, FFTW, BandedMatrices, SemiseparableMatrices, ToeplitzMatrices, SparseArrays, Printf
include("BoundaryConditions.jl")
include("AlmostBandedPreconditioner.jl")
include("UltraS.jl")
include("Utilities.jl")
include("Chebtech2.jl")
include("JacSolver.jl")
include("LinearSolver.jl")
include("TRC.jl")
include("TRD.jl")
include("LSB.jl")
include("PathFollowing.jl")