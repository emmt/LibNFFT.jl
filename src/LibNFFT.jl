module LibNFFT

export
    NFCT,
    NFFT,
    NFST,
    apply_adjoint!,
    apply_adjoint,
    apply_direct!,
    apply_direct

# Types to identify raw plan structures parameterized by the floating-point
# type.  These must be defined before including "../deps/deps.jl".
abstract type _nfft_plan{T<:AbstractFloat} end
abstract type _nfct_plan{T<:AbstractFloat} end
abstract type _nfst_plan{T<:AbstractFloat} end

isfile(joinpath(@__DIR__, "..", "deps", "deps.jl")) ||
    error("LibNFFT not properly installed.  Please run Pkg.build(\"LibNFFT\")")
include(joinpath("..", "deps", "deps.jl"))

import FFTW

const OPENMP = false

const DEFAULT_FFTW_FLAGS = FFTW.ESTIMATE
const MINIMAL_FFTW_FLAGS = FFTW.DESTROY_INPUT
const ALLOWED_FFTW_FLAGS = (FFTW.ESTIMATE | FFTW.MEASURE |
                            FFTW.PATIENT  | FFTW.EXHAUSTIVE)

const DEFAULT_NFFT_FLAGS = (MALLOC_X  | MALLOC_F_HAT | MALLOC_F |
                            PRE_PHI_HUT | PRE_PSI)
const MINIMAL_NFFT_FLAGS = (_FFTW_INIT | _FFT_OUT_OF_PLACE  |
                            (OPENMP ? _OMP_BLOCKWISE_ADJOINT : 0))
const ALLOWED_NFFT_FLAGS = (MALLOC_X    | MALLOC_F | MALLOC_F_HAT |
                            PRE_PHI_HUT | FG_PSI   | PRE_LIN_PSI  |
                            PRE_FG_PSI  | PRE_PSI  | PRE_FULL_PSI |
                            SORT_NODES)

const Floats = Union{Cfloat,Cdouble}
const RealOrComplex{T} = Union{T,Complex{T}}
const SinglePrecision{T} = RealOrComplex{Cfloat}
const DoublePrecision{T} = RealOrComplex{Cdouble}

"""

`LibNFFT.NullPointerError()` is the exception thrown when a pointer has an
invalid `NULL` value.

"""
struct NullPointerError <: Exception end

"""
    LibNFFT.version(T) -> (major, minor, path)

yields the version of the NFFT library for floating-point type `T`.

""" version


"""

Abstract type `LibNFFT.Plan{T,d,E}` is the super-type of all plans implementing
one of the transforms provided by the NFFT library.

This type is parameterized by `T` the floating-point type for the computations,
`d` and `E` the number of dimensions and the type of the elements of the input
arrays.  Parameter `E` is either `T` or `Complex{T}`.  Because, `E` depends on
`T`, `T` must be specified first in the signature.

"""
abstract type Plan{T,d,E} end

# `NFFT{T,d,E}` (nonequispaced fast Fourier transform).
mutable struct NFFT{T<:Floats,d,E<:Union{T,Complex{T}}} <: Plan{T,d,E}
    ptr::Ptr{_nfft_plan{T}}    # plan to compute NFFT
    f_hat::Array{Complex{T},d} # input samples
    f::Vector{Complex{T}}      # result of transform
    x::Matrix{T}               # nodes
    ready::Bool

    # The inner constructor takes the same arguments as the nfft_init_guru
    # function except that it allocates the structure.
    function NFFT{T,d,E}(inpdims::NTuple{d,Int}, # input dimensions (N)
                         M::Int,                 # number of nodes (M)
                         ovrdims::NTuple{d,Int}, # oversampled dimensions (n)
                         m::Int,                 # window size
                         flags::Cuint,           # NFFT flags
                         fftw_flags::Cuint       # FFTW flags
                         ) where {T<:Floats,d,E<:Union{T,Complex{T}}}
        # Create object and attach a finalizer to it.
        ptr, f_hat, f, x = _create(_nfft_plan{T}, inpdims, M, ovrdims, m,
                                   flags, fftw_flags)
        return finalizer(finalize, new{T,d,E}(ptr, f_hat, f, x, false))
    end
end

# `NFCT{T,d}` (nonequispaced fast cosine transform).
mutable struct NFCT{T<:Floats,d} <: Plan{T,d,T}
    ptr::Ptr{_nfct_plan{T}}   # plan to compute NFCT
    f_hat::Array{T,d}         # input samples
    f::Vector{T}              # result of transform
    x::Matrix{T}              # nodes
    ready::Bool

    # The inner constructor takes the same arguments as the nfct_init_guru
    # function except that it allocates the structure.
    function NFCT{T,d}(inpdims::NTuple{d,Int}, # input dimensions (N)
                       M::Int,                 # number of nodes (M)
                       ovrdims::NTuple{d,Int}, # oversampled dimensions (n)
                       m::Int,                 # window size
                       flags::Cuint,           # NFCT flags
                       fftw_flags::Cuint       # FFTW flags
                       ) where {T<:Floats,d,E<:Union{T,Complex{T}}}
        # Create object and attach a finalizer to it.
        ptr, f_hat, f, x = _create(_nfct_plan{T}, inpdims, M, ovrdims, m,
                                   flags, fftw_flags)
        return finalizer(finalize, new{T,d}(ptr, f_hat, f, x, false))
    end
end

# `NFST{T,d}` (nonequispaced fast sine transform).
mutable struct NFST{T<:Floats,d} <: Plan{T,d,T}
    ptr::Ptr{_nfst_plan{T}}   # plan to compute NFST
    f_hat::Array{T,d}         # input samples
    f::Vector{T}              # result of transform
    x::Matrix{T}              # nodes
    ready::Bool

    # The inner constructor takes the same arguments as the nfst_init_guru
    # function except that it allocates the structure.
    function NFST{T,d}(inpdims::NTuple{d,Int}, # input dimensions (N)
                       M::Int,                 # number of nodes (M)
                       ovrdims::NTuple{d,Int}, # oversampled dimensions (n)
                       m::Int,                 # window size
                       flags::Cuint,           # NFST flags
                       fftw_flags::Cuint       # FFTW flags
                       ) where {T<:Floats,d,E<:Union{T,Complex{T}}}
        # Create object and attach a finalizer to it.
        ptr, f_hat, f, x = _create(_nfst_plan{T}, inpdims, M, ovrdims, m,
                                   flags, fftw_flags)
        return finalizer(finalize, new{T,d}(ptr, f_hat, f, x, false))
    end
end

const BasicPlans{T,d,E} = Union{NFFT{T,d,E},NFCT{T,d},NFST{T,d}}

function _create(::Type{T},
                 inpdims::NTuple{d,Int}, # input dimensions (N)
                 M::Int,                 # number of nodes (M)
                 ovrdims::NTuple{d,Int}, # oversampled dimensions (n)
                 m::Int,                 # window size
                 flags::Cuint,           # NFFT flags
                 fftw_flags::Cuint       # FFTW flags
                 ) where {T<:Union{_nfft_plan,_nfct_plan,_nfst_plan},
                          d,E<:Union{T,Complex{T}}}
    # Notations:
    # `d` is the number of dimensions (the rank);
    # `N = [N1,N2,...,Nd]` are the dimensions of the result;
    # `M` are the number of nodes;
    # `n = [n1,n2,...,nd]` are the oversampled dimensions for the FFT;
    # `m` is the size of the window.

    # Check rank, number of nodes and window size.
    d ≥ 1 || argument_error("invalid number of dimensions")
    M ≥ 1 || argument_error("invalid number of nodes")
    m ≥ 2 || argument_error("window size must be at least 2")

    # Check dimensions, convert them to C-int integer type and reorder
    # them in the same order as expected by FFTW/NFFT.
    N = Vector{Cint}(undef, d)
    n = Vector{Cint}(undef, d)
    for i in 1:d
        inpdim::Int = inpdims[i]
        ovrdim::Int = ovrdims[i]
        inpdim < 1 && argument_error("invalid dimension(s)")
        ovrdim ≤ inpdim && argument_error("oversampled dimension(s) too small")
        isodd(inpdim) && argument_error("dimensions must be even numbers")
        N[d + 1 - i] = inpdim
        n[d + 1 - i] = ovrdim
    end

    # Create the plan and initialize it using the "guru" interface.
    ptr = _allocate(T)
    _init_guru(ptr, d, N, M, n, m, flags, fftw_flags)

    # Retrieve the work arrays.
    f_hat = _init_array(_pointer_to_f_hat(ptr), inpdims, (flags & MALLOC_F_HAT) != 0)
    f = _init_array(_pointer_to_f(ptr), (Int(M),), (flags & MALLOC_F) != 0)
    x = _init_array(_pointer_to_x(ptr), (d, Int(M)), (flags & MALLOC_X) != 0)

    return (ptr, f_hat, f, x)
end

function _init_array(ptr::Ptr{Ptr{T}},
                     dims::NTuple{N,Int},
                     wrap::Bool) :: Array{T,N} where {T,N}
    if wrap
        return unsafe_wrap(Array, unsafe_load(ptr), dims; own = false)
    else
        A = Array{T,N}(undef, dims)
        unsafe_store!(ptr, pointer(A))
        return A
    end
end

# Outer constructors.
function NFFT(M::Integer,                          # number of nodes
              E::Union{Type{T}, Type{Complex{T}}}, # input element type
              dims::NTuple{d,Integer};             # input dimensions
              m::Integer = 6,                      # window size
              sigma::Real = 2.0,                   # oversampling factor
              flags::Integer = DEFAULT_NFFT_FLAGS,
              fftw_flags::Integer = DEFAULT_FFTW_FLAGS) where {T<:Floats,d}
    # Create NFFT plan with minimal set of options enforced.
    inpdims, ovrdims = _fix_dims(NFFT, dims, sigma)
    return NFFT{T,d,E}(inpdims, Int(M), ovrdims, Int(m),
                       Cuint(flags | MINIMAL_NFFT_FLAGS),
                       Cuint(fftw_flags | MINIMAL_FFTW_FLAGS))
end

function NFFT(x::AbstractMatrix{<:AbstractFloat},  # coordinates of nodes
              E::Union{Type{T}, Type{Complex{T}}}, # input element type
              dims::NTuple{d,Integer};             # input dimensions
              kwds...) where {T<:Floats,d}
    size(x, 1) == d || dimension_mismatch("bad first dimension for node coordinates")
    M = size(x, 2)
    return set_x!(NFFT(M, E, dims; kwds...), x)
end

function NFCT(M::Integer,                          # number of nodes
              ::Type{T},                           # floating-point type
              dims::NTuple{d,Integer};             # input dimensions
              m::Integer = 6,                      # window size
              sigma::Real = 2.0,                   # oversampling factor
              flags::Integer = DEFAULT_NFFT_FLAGS,
              fftw_flags::Integer = DEFAULT_FFTW_FLAGS) where {T<:Floats,d}
    # Create NFCT plan with minimal set of options enforced.
    inpdims, ovrdims = _fix_dims(NFCT, dims, sigma)
    return NFCT{T,d}(inpdims, Int(M), ovrdims, Int(m),
                     Cuint(flags | MINIMAL_NFFT_FLAGS),
                     Cuint(fftw_flags | MINIMAL_FFTW_FLAGS))
end

function NFCT(x::AbstractMatrix{<:AbstractFloat},  # coordinates of nodes
              ::Type{T},                           # floating-point type
              dims::NTuple{d,Integer};             # input dimensions
              kwds...) where {T<:Floats,d}
    size(x, 1) == d || dimension_mismatch("bad first dimension for node coordinates")
    M = size(x, 2)
    return set_x!(NFCT(M, T, dims; kwds...), x)
end

function NFST(M::Integer,                          # number of nodes
              ::Type{T},                           # floating-point type
              dims::NTuple{d,Integer};             # input dimensions
              m::Integer = 6,                      # window size
              sigma::Real = 2.0,                   # oversampling factor
              flags::Integer = DEFAULT_NFFT_FLAGS,
              fftw_flags::Integer = DEFAULT_FFTW_FLAGS) where {T<:Floats,d}
    # Create NFST plan with minimal set of options enforced.
    inpdims, ovrdims = _fix_dims(NFST, dims, sigma)
    return NFST{T,d}(inpdims, Int(M), ovrdims, Int(m),
                     Cuint(flags | MINIMAL_NFFT_FLAGS),
                     Cuint(fftw_flags | MINIMAL_FFTW_FLAGS))
end

function NFST(x::AbstractMatrix{<:AbstractFloat},  # coordinates of nodes
              ::Type{T},                           # floating-point type
              dims::NTuple{d,Integer};             # input dimensions
              kwds...) where {T<:Floats,d}
    size(x, 1) == d || dimension_mismatch("bad first dimension for node coordinates")
    M = size(x, 2)
    return set_x!(NFST(M, T, dims; kwds...), x)
end

function finalize(ths::Union{NFFT{T,d},NFCT{T,d},NFST{T,d}}) where {T,d}
    if (ptr = ths.ptr) != C_NULL
        ths.ptr = C_NULL # to avoid finalize more than once
        _finalize(ptr)
        _free(ptr)
    end
    ths.f_hat = Array{Complex{T},d}(undef, ntuple(x->0, Val(d)))
    ths.f = Vector{Complex{T}}(undef, 0)
    ths.x = Matrix{T}(undef, d, 0)
    ths.ready = false
    nothing
end

function _fix_dims(::Type{<:BasicPlans},
                   dims::Tuple{Vararg{Integer}}, sigma::Real)
    # Get good dimensions for the NFFT suitable fot the FFT (hence powers of 2,
    # 3 or 5) and are larger of equal `sigma` times the dimensions in `dims`.
    sigma < 1 && argument_error("oversampling factor must be at least one")
    ovrdims = map(dim -> nextprod([2,3,5], ceil(Int, sigma*dim)), dims)
    return map(Int, dims), ovrdims
end

# Yields the number of nodes.
get_M(ths::BasicPlans) = size(ths.x, 2)

# Yields the number of dimensions of the input.
get_d(ths::Plan{T,d}) where {T,d} = d

get_flags(ths::BasicPlans) = get_flags(ths.ptr)

get_fftw_flags(ths::BasicPlans) = get_fftw_flags(ths.ptr)

isready(ths::BasicPlans) = ths.ready

argument_error(x) = throw(ArgumentError(x))
dimension_mismatch(x) = throw(DimensionMismatch(x))

Base.show(io::IO, ::MIME"text/plain", ths::NFFT{T,d,E}) where {T,d,E} =
    print(io, "LibNFFT.NFFT{$T,$d,$E} ",
          "input: $E $(size(ths.f_hat)), ",
          "output: Complex{$T} $(size(ths.f))")

Base.show(io::IO, ::MIME"text/plain", ths::NFCT{T,d}) where {T,d} =
    print(io, "LibNFFT.NFCT{$T,$d} ",
          "input: $T $(size(ths.f_hat)), ",
          "output: $T $(size(ths.f))")

Base.show(io::IO, ::MIME"text/plain", ths::NFST{T,d}) where {T,d} =
    print(io, "LibNFFT.NFST{$T,$d} ",
          "input: $T $(size(ths.f_hat)), ",
          "output: $T $(size(ths.f))")

"""
    LibNFFT.get_x(ths) -> x

yields the coordinates of the nodes in NFFT plan `ths`.  The result `x` is such
that `x[t,j]` is `t`-th coordinate of `j`-th node.

If the node coordinates have not yet been set in `ths`, all coordinates in the
result have `NaN` values.

The result `x` is a new array as coordinates are stored in reverse order by
NFFT (compared to Julia).  To re-use storage, the in-place version can be
called instead:

    LibNFFT.get_x!(x, ths) -> x

overwrites `x` with the coordinates of the nodes in NFFT plan `ths`.

See also [`NFFT`](@ref), [`NFCT`](@ref), [`NFST`](@ref),
[`LibNFFT.set_x!`](@ref).

"""
get_x(ths::NFFT{T}) where{T} = get_x!(Matrix{T}(undef, size(ths.x)), ths)

function get_x!(x::AbstractMatrix{<:AbstractFloat},
                ths::NFFT{T,d,E}) where {T,d,E}
    # Check dimensions.
    src = ths.x
    dims = size(src)
    dims[1] == d || error("corrupted NFFT structure")
    M = dims[2]
    Base.has_offset_axes(x) &&
        error("array of node coordinates has non-standard indexing")
    size(x) == dims ||
        dimension_mismatch("incompatible dimensions of node coordinates")

    # Copy node coordinates (dimensions are in reverse order in NFFT).
    if ths.ready
        @inbounds begin
            for j in 1:M
                @simd for i in 1:d
                    x[i,j] = src[(d+1)-i,j]
                end
            end
        end
    else
        fill!(x, NaN)
    end
    return x
end

@doc @doc(get_x) get_x!

"""
    LibNFFT.set_x!(ths, x) -> ths

sets the coordinates of the nodes in NFFT (or NFCT or NFST) plan `ths`.
Argument `x` is such that `x[t,j]` is `t`-th coordinate of `j`-th node.  Thus
`x` must be an array of dimensions `(d,M)` with `d` the number of dimensions of
the input argument of the direct transform and `M` the number of nodes.  All
node coordinates must be in the semi-open range `[-1/2,+1/2)`.

See also [`NFFT`](@ref), [`NFCT`](@ref), [`NFST`](@ref),
[`LibNFFT.get_x`](@ref).

"""
function set_x!(ths::BasicPlans{<:AbstractFloat,d},
                x::AbstractMatrix{T}) where {T<:AbstractFloat,d}
    # Check dimensions.  Node coordinates must be an array of dimensions
    # `(d,M)` with `d` the number of dimensions of the input argument of the
    # direct transform and `M` the number of nodes.
    dest = ths.x
    dims = size(dest)
    dims[1] == d || error("corrupted NFFT structure")
    M = dims[2]
    Base.has_offset_axes(x) &&
        error("array of node coordinates has non-standard indexing")
    size(x) == dims ||
        dimension_mismatch("incompatible dimensions of node coordinates")

    # Check values of node coordinates.
    cmax = one(T)/2
    cmin = -cmax
    @inbounds begin
        for j in 1:M
            for i in 1:d
                c = x[i,j]
                isnan(c) && error("invalid NaN node coordinate")
                isinf(c) && error("invalid infinite node coordinate")
                cmin ≤ c < cmax ||
                    error("node coordinate(s) out of range [-0.5,0.5)")
            end
        end
    end

    # Copy node coordinates (dimensions are in reverse order in NFFT).
    @inbounds begin
        for j in 1:M
            @simd for i in 1:d
                dest[(d+1)-i,j] = x[i,j]
            end
        end
    end

    if ths.ptr != C_NULL
        # Pre-compute PSI if needed.
        if isa(ths, NFFT)
            if (get_flags(ths) & (PRE_LIN_PSI|PRE_FG_PSI|PRE_PSI|PRE_FULL_PSI)) != 0
                _precompute_one_psi(ths.ptr)
            end
        else
            if (get_flags(ths) & PRE_PSI) != 0
                _precompute_psi(ths.ptr)
            end
        end
        ths.ready = true
    else
        ths.ready = false
    end
    return ths
end

# Implement behavior of the NFFT operator.
#
# It is not advisable to provide a single method, say
# `apply!(dst,ths,src,adj::Bool=false)` to apply either the direct (`adj is
# false) or the adjoint (`adj` is true) because this would break type
# stability.

function apply_direct!(dst, ths::BasicPlans, src)
    isready(ths) || error("the node coordinates must have been set")
    _copyto!(ths.f_hat, src)
    _apply_direct(ths.ptr)
    return _copyto!(dst, ths.f)
end

function apply_adjoint!(dst, ths::BasicPlans, src)
    isready(ths) || error("the node coordinates must have been set")
    _copyto!(ths.f, src)
    _apply_adjoint!(ths.ptr)
    return _copyto!(dst, ths.f_hat)
end

apply_direct(ths::Plan, src) =
    apply_direct!(create_f(ths), ths, src)

apply_adjoint(ths::Plan, src) =
    apply_adjoint!(create_f_hat(ths), ths, src)

create_f(ths::NFFT{T,d,E}) where {T,d,E} =
    Vector{Complex{T}}(undef, size(ths.f))

create_f(ths::Union{NFCT{T,d},NFST{T,d}}) where {T,d} =
    Vector{T}(undef, size(ths.f))

create_f_hat(ths::NFFT{T,d,E}) where {T,d,E} =
    Array{E,d}(undef, size(ths.f_hat))

create_f_hat(ths::Union{NFCT{T,d},NFST{T,d}}) where {T,d} =
    Array{T,d}(undef, size(ths.f_hat))

"""
    unsafe_load_field(ptr, T, off)

yields the value of a field of type `T` at relative offset `off` (in bytes) in
a structure whose base address is `ptr`.  A result of type `T` is returned.  If
`ptr` is NULL, a `LibNFFT.NullPointerError` exception is thrown.

"""
unsafe_load_field(ptr::Ptr, ::Type{T}, off::Integer) where {T} =
    (ptr == C_NULL ? throw(NullPointerError()) :
     unsafe_load(Ptr{T}(ptr + off)))

# Allocate dynamic memory:
#     ptr = _allocate(nbytes)
#     ptr = _allocate(T, num=1)
# then:
#     _free(ptr)
#
function _allocate(size::Integer)
    ptr = Libc.malloc(size)
    ptr != C_NULL || thrown(OutOfMemoryError())
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
          ptr, 0, size)
    return ptr
end

_allocate(::Type{T}, n::Integer=1) where {T} =
    Ptr{T}(_allocate(n*sizeof(T)))

_free(ptr::Ptr) =
    ptr != C_NULL && Libc.free(ptr)

"""
    indices(A, ...)

yields an iterable object for visiting each index of its arguments.  The is is
the same as `eachindex` except that arguments must be abstract arrays and that
it is checked that arguments have the same dimensions and have standard
indexing.

"""
@inline function indices(A::AbstractArray)
    Base.has_offset_axes(A) &&
        error("argument(s) have non-standard indexing")
    return eachindex(A)
end
@inline function indices(A::AbstractArray, B::AbstractArray)
    Base.has_offset_axes(A, B) &&
        error("argument(s) have non-standard indexing")
    size(A) == size(B) ||
        dimension_mismatch("arguments have not smae dimensions")
    return eachindex(A, B)
end
@inline function indices(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    Base.has_offset_axes(A, B, C) &&
        error("argument(s) have non-standard indexing")
    size(A) == size(B) == size(C) ||
        dimension_mismatch("arguments have not smae dimensions")
    return eachindex(A, B, C)
end

"""
    _copyto!(dst, src) -> dst

copies the values of the source `src` into the destination `dst`.  An error is
thrown if the source and destination do not have the same dimensions oir have
non-standard indexing.  The types of their elements may be different.  The
`copyto!` method of Julia does not impose that the dimensions be the same and
may be slower than `_copyto!`.  For instance, copying from a complex array to a
real array is about 5 times slower because of the checking of the imaginary
part (here it is jut ignored).

"""
function _copyto!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    @inbounds @simd for i in indices(A, B)
        dst[i] = src[i]
    end
    return dst
end

function _copyto!(dst::Array{<:Real,N}, src::Array{<:Real,N}) where {N}
    @inbounds @simd for i in indices(A, B)
        dst[i] = src[i]
    end
    return dst
end

function _copyto!(dst::Array{<:Complex,N}, src::Array{<:Complex,N}) where {N}
    @inbounds @simd for i in indices(A, B)
        dst[i] = src[i]
    end
    return dst
end

function _copyto!(dst::Array{<:Complex,N}, src::Array{<:Real,N}) where {N}
    @inbounds @simd for i in indices(A, B)
        dst[i] = src[i]
    end
    return dst
end

function _copyto!(dst::Array{<:Real,N}, src::Array{<:Complex,N}) where {N}
    @inbounds @simd for i in indices(A, B)
        dst[i] = src[i].re
    end
    return dst
end

end # module
