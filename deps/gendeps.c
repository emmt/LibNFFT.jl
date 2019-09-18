/*
 * gendeps.c --
 *
 * Generate constants definitions for Julia.
 *
 *------------------------------------------------------------------------------
 *
 * This file is part of LibNFFT.jl released under the MIT "expat" license.
 * Copyright (C) 2016-2018, Éric Thiébaut (https://github.com/emmt/LibNFFT.jl).
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <nfft3.h>

#define CHOICE(T,f,d,q) (sizeof(T) == sizeof(float) ? (f) :            \
                         (sizeof(T) == sizeof(double) ? (d) : (q)))

#define FLT_SFX "f"
#define DBL_SFX ""
#define QUD_SFX "l"

#define SUFFIX(T) CHOICE(T, FLT_SFX, DBL_SFX, QUD_SFX)

#define LIBRARY(T)                                                      \
    CHOICE(T, "libnfft" FLT_SFX, "libnfft" DBL_SFX, "libnfft" QUD_SFX)


/*
 * Determine the offset of a field in a structure.
 */
#define OFFSET_OF(type, field) (int)((char*)&((type*)0)->field - (char*)0)

/*
 * Determine whether an integer type is signed.
 */
#define IS_SIGNED(type)        ((type)(~(type)0) < (type)0)

/*
 * Set all the bits of an L-value.
 */
#define SET_ALL_BITS(lval) lval = 0; lval = ~lval

/*
 * Define a Julia alias for a C integer, given an L-value of the corresponding
 * type.
 */
#define DEF_TYPEOF_LVALUE(name, lval)           \
    do {                                        \
        SET_ALL_BITS(lval);                     \
        printf("const _typeof_%s = %sInt%u\n",  \
               name, (lval < 0 ? "" : "U"),     \
               (unsigned)(8*sizeof(lval)));     \
                                                \
    } while (0)

/*
 * Define a Julia alias for a C integer, given its type (`space` is used for
 * alignment).
 */
#define DEF_TYPEOF_TYPE(type, space)                    \
    do {                                                \
        type lval;                                      \
        SET_ALL_BITS(lval);                             \
        printf("const _typeof_%s%s = %sInt%u\n",        \
               #type, space, (lval < 0 ? "" : "U"),     \
               (unsigned)(8*sizeof(lval)));             \
                                                        \
    } while (0)

#define DEF_TYPEOF_FIELD(type, field)                                   \
    do {                                                                \
        type obj;                                                       \
        SET_ALL_BITS(obj.field);                                        \
        printf("const _typeof_" #type "_" #field " = %sInt%u\n",        \
               (obj.field < 0 ? "" : "U"),                              \
               (unsigned)(8*sizeof(obj.field)));                        \
                                                                        \
    } while (0)

#define TYPEOF_INTEGER_LVALUE(dest, lval)                       \
    do {                                                        \
        SET_ALL_BITS(lval);                                     \
        dest = ((lval < 0 ? SIGNED : UNSIGNED) | sizeof(lval)); \
    } while (0)

#define TYPEOF_INTEGER_TYPE(dest, type)         \
    do {                                        \
        type __temp;                            \
        TYPEOF_INTEGER_LVALUE(dest, __temp);    \
    } while (0)

#define TYPEOF_INTEGER_FIELD(dest, type, field)         \
    do {                                                \
        type __temp;                                    \
        TYPEOF_INTEGER_LVALUE(dest, __temp.field);      \
    } while (0)


/* Lowest bits are used to store the size (in bytes). */
#define  SIGNED   (1 << 24) /* signed integer */
#define  UNSIGNED (1 << 25) /* unsigned integer */
#define  FLOAT    (1 << 26) /* floating-point */
#define  COMPLEX  (1 << 27) /* complex */
#define  POINTER  (1 << 28) /* pointer */

static const char*
typename(char* buf, int type)
{
    char* ptr = buf;
    int size = (type & 0xffffff);
    *ptr = 0;
    if ((type & POINTER) != 0) {
        strcat(ptr, "Ptr{");
        ptr += strlen(ptr);
    }
    switch (type & (SIGNED|UNSIGNED|FLOAT|COMPLEX)) {
    case SIGNED:
        sprintf(ptr, "Int%d", 8*size);
        break;
    case UNSIGNED:
        sprintf(ptr, "UInt%d", 8*size);
        break;
    case FLOAT:
        sprintf(ptr, "Float%d", 8*size);
        break;
    case COMPLEX:
        sprintf(ptr, "Complex{Float%d}", 4*size);
        break;
    case COMPLEX|FLOAT:
        sprintf(ptr, "Complex{Float%d}", 8*size);
        break;
    case 0:
        strcat(ptr, "Cvoid");
        break;
    default:
        fprintf(stderr, "invalid type specification\n");
        exit(1);
    }
    if ((type & POINTER) != 0) {
        strcat(ptr, "}");
    }
    return buf;
}

static void
address_of(FILE* output, const char* funcname,
           const char* argtype, const char* ptrtype, long offset)
{
    fprintf(output,
            "%s(ptr::%s) = \n"
            "    Ptr{%s}(ptr + %ld)\n", funcname, argtype, ptrtype, offset);
}

static void
get_field(FILE* output, const char* funcname, const char* argtype,
          const char* fieldtype, long fieldoffset)
{
    fprintf(output, "%s(ptr::%s) = unsafe_load_field(ptr, %s, %ld)\n",
            funcname, argtype, fieldtype, fieldoffset);
}

static void
pointer_to(FILE* output, const char* funcname, const char* argtype,
          const char* fieldtype, long fieldoffset)
{
    fprintf(output, "%s(ptr::%s) = Ptr{%s}(ptr + %ld)\n",
            funcname, argtype, fieldtype, fieldoffset);
}

/*
 * Define a Julia constant with the offset (in bytes) of a field of a
 * C-structure.
 */
#define DEF_OFFSETOF(type, field)                                       \
    fprintf(output, "const _offsetof_" #type "_" #field " = %ld\n",     \
            (long)OFFSET_OF(type, field))

int main(int argc, char* argv[])
{
    FILE* output = stdout;
    const char* quote;
    char buffer[100];
    fprintf(output,
            "# This file has been automatically generated, do not edit it but rather run\n"
            "# `make deps.jl` from the shell or execute `Pkg.build(\"LibNFFT\") from julia.\n");

    fprintf(output,
            "\n"
            "# Path to the NFFT dynamic libraries (single and double precision versions).\n");
    quote = (LIB_NFFTF_DLL[0] == '"' ? "" : "\"");
    fprintf(output, "const %s = %s%s%s\n", LIBRARY(float), quote, LIB_NFFTF_DLL, quote);
    quote = (LIB_NFFT_DLL[0] == '"' ? "" : "\"");
    fprintf(output, "const %s = %s%s%s\n", LIBRARY(double), quote, LIB_NFFT_DLL, quote);

    {
        unsigned int major, minor, patch;
        nfft_get_version(&major, &minor, &patch);
        fprintf(output,
                "\n# NFFT version for which package has been built.\n"
                "NFFT_VERSION = v\"%u.%u.%u\"\n", major, minor, patch);
    }

#define DEF_FLAG1(a)   fprintf(output, "const %22s = 0x%08x\n", #a, a)
#define DEF_FLAG2(a,b) fprintf(output, "const %22s = 0x%08x\n", #a, b)

    fprintf(output, "\n# NFFT public flags.\n");
    DEF_FLAG1(PRE_PHI_HUT);
    DEF_FLAG1(FG_PSI);
    DEF_FLAG1(PRE_LIN_PSI);
    DEF_FLAG1(PRE_FG_PSI);
    DEF_FLAG1(PRE_PSI);
    DEF_FLAG1(PRE_FULL_PSI);
    DEF_FLAG2(SORT_NODES, NFFT_SORT_NODES);
    DEF_FLAG1(MALLOC_X);
    DEF_FLAG1(MALLOC_F);
    DEF_FLAG1(MALLOC_F_HAT);

    fprintf(output, "\n# NFFT private flags.\n");
    DEF_FLAG2(_FFTW_INIT, FFTW_INIT);
    DEF_FLAG2(_FFT_OUT_OF_PLACE, FFT_OUT_OF_PLACE);
    DEF_FLAG2(_OMP_BLOCKWISE_ADJOINT, NFFT_OMP_BLOCKWISE_ADJOINT);

#undef DEF_FLAG1
#undef DEF_FLAG2

#define GET_INTEGER_FIELD(argtype, datatype, field)                     \
    do {                                                                \
        int __typeid;                                                   \
        char __typename[100];                                           \
        TYPEOF_INTEGER_FIELD(__typeid, datatype, field);                \
        get_field(output, "get_" #field, argtype,                       \
                  typename(__typename, __typeid),                       \
                  OFFSET_OF(datatype, field));                          \
    } while (0)

#define POINTER_TO(argtype, datatype, field, elemtype)                  \
    do {                                                                \
        pointer_to(output, "_pointer_to_" #field, argtype,              \
                   elemtype, OFFSET_OF(datatype, field));               \
    } while (0)

    /* Get version. */
    fprintf(output, "\n# Retrieve library version number.\n");
#define METHODS(real)                                                                    \
    do {                                                                                 \
        fprintf(output,                                                                  \
                "function version(::Type{C%s})\n"                                        \
                "    major = Ref{Cuint}()\n"                                             \
                "    minor = Ref{Cuint}()\n"                                             \
                "    patch = Ref{Cuint}()\n"                                             \
                "    ccall((:nfft%s_get_version, %s), Cvoid,\n"                          \
                "          (Ptr{Cuint},Ptr{Cuint},Ptr{Cuint}), major, minor, patch)\n"   \
                "    return (Int(major[]), Int(minor[]), Int(patch[]))\n"                \
                "end\n", #real, SUFFIX(real), LIBRARY(real));                            \
    } while (0)
    METHODS(float);
    METHODS(double);
#undef METHODS

    /* Constants and methods for NFFT (nonequispaced fast Fourier
       transform). */
#define METHODS(plan, real)                                                              \
    do {                                                                                 \
        const char* argtype = "Ptr{_nfft_plan{C" #real "}}";                             \
                                                                                         \
        fprintf(output,                                                                  \
                "\n# Methods for the C structure `" #plan "`.\n");                       \
        fprintf(output, "Base.sizeof(::Type{%s}) = %ld\n",                               \
                "_nfft_plan{C" #real "}", (long)sizeof(plan));                           \
        POINTER_TO(argtype, plan, f_hat, "Ptr{Complex{C" #real "}}");                    \
        POINTER_TO(argtype, plan, f,     "Ptr{Complex{C" #real "}}");                    \
        POINTER_TO(argtype, plan, x,     "Ptr{C" #real "}");                             \
        GET_INTEGER_FIELD(argtype, plan, flags);                                         \
        GET_INTEGER_FIELD(argtype, plan, fftw_flags);                                    \
        fprintf(output,                                                                  \
                "function _init_guru(ptr::Ptr{_nfft_plan{C%s}}, d::Integer,\n"           \
                "                    N::Vector{Cint}, M::Integer, n::Vector{Cint},\n"    \
                "                    m::Integer, flags::Integer, fftw_flags::Integer)\n" \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfft%s_init_guru, %s), Cvoid,\n"                            \
                "          (Ptr{_nfft_plan{C%s}}, Cint, Ptr{Cint}, Cint, Ptr{Cint},\n"   \
                "           Cint, Cuint, Cuint), ptr, d, N, M, n, m, flags,\n"           \
                "          fftw_flags)\n"                                                \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _finalize(ptr::Ptr{_nfft_plan{C%s}})\n"                        \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfft%s_finalize, %s), Cvoid,\n"                             \
                "          (Ptr{_nfft_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _precompute_one_psi(ptr::Ptr{_nfft_plan{C%s}})\n"              \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfft%s_precompute_one_psi, %s), Cvoid,\n"                   \
                "          (Ptr{_nfft_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _apply_direct(ptr::Ptr{_nfft_plan{C%s}})\n"                    \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfft%s_trafo, %s), Cvoid,\n"                                \
                "          (Ptr{_nfft_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _apply_adjoint(ptr::Ptr{_nfft_plan{C%s}})\n"                   \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfft%s_adjoint, %s), Cvoid,\n"                              \
                "          (Ptr{_nfft_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
    } while (0)
    METHODS(nfftf_plan, float);
    METHODS(nfft_plan, double);
#undef METHODS


    /* Constants and methods for NFCT (nonequispaced fast cosine
       transform). */
#define METHODS(plan, real)                                                              \
    do {                                                                                 \
        const char* argtype = "Ptr{_nfct_plan{C" #real "}}";                             \
                                                                                         \
        fprintf(output,                                                                  \
                "\n# Methods for the C structure `" #plan "`.\n");                       \
        fprintf(output, "Base.sizeof(::Type{%s}) = %ld\n",                               \
                "_nfct_plan{C" #real "}", (long)sizeof(plan));                           \
        POINTER_TO(argtype, plan, f_hat, "Ptr{C" #real "}");                             \
        POINTER_TO(argtype, plan, f,     "Ptr{C" #real "}");                             \
        POINTER_TO(argtype, plan, x,     "Ptr{C" #real "}");                             \
        GET_INTEGER_FIELD(argtype, plan, flags);                                         \
        GET_INTEGER_FIELD(argtype, plan, fftw_flags);                                    \
        fprintf(output,                                                                  \
                "function _init_guru(ptr::Ptr{_nfct_plan{C%s}}, d::Integer,\n"           \
                "                    N::Vector{Cint}, M::Integer, n::Vector{Cint},\n"    \
                "                    m::Integer, flags::Integer, fftw_flags::Integer)\n" \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfct%s_init_guru, %s), Cvoid,\n"                            \
                "          (Ptr{_nfct_plan{C%s}}, Cint, Ptr{Cint}, Cint, Ptr{Cint},\n"   \
                "           Cint, Cuint, Cuint), ptr, d, N, M, n, m, flags,\n"           \
                "          fftw_flags)\n"                                                \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _finalize(ptr::Ptr{_nfct_plan{C%s}})\n"                        \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfct%s_finalize, %s), Cvoid,\n"                             \
                "          (Ptr{_nfct_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _precompute_psi(ptr::Ptr{_nfct_plan{C%s}})\n"                  \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfct%s_precompute_psi, %s), Cvoid,\n"                       \
                "          (Ptr{_nfct_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _apply_direct(ptr::Ptr{_nfct_plan{C%s}})\n"                    \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfct%s_trafo, %s), Cvoid,\n"                                \
                "          (Ptr{_nfct_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _apply_adjoint(ptr::Ptr{_nfct_plan{C%s}})\n"                   \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfct%s_adjoint, %s), Cvoid,\n"                              \
                "          (Ptr{_nfct_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
    } while (0)
    METHODS(nfctf_plan, float);
    METHODS(nfct_plan, double);
#undef METHODS


    /* Constants and methods for NFST (nonequispaced fast sine transform). */
#define METHODS(plan, real)                                             \
    do {                                                                \
        const char* argtype = "Ptr{_nfst_plan{C" #real "}}";            \
                                                                        \
        fprintf(output,                                                 \
                "\n# Methods for the C structure `" #plan "`.\n");      \
        fprintf(output, "Base.sizeof(::Type{%s}) = %ld\n",              \
                "_nfst_plan{C" #real "}", (long)sizeof(plan));          \
        POINTER_TO(argtype, plan, f_hat, "Ptr{C" #real "}");            \
        POINTER_TO(argtype, plan, f,     "Ptr{C" #real "}");            \
        POINTER_TO(argtype, plan, x,     "Ptr{C" #real "}");            \
        GET_INTEGER_FIELD(argtype, plan, flags);                        \
        GET_INTEGER_FIELD(argtype, plan, fftw_flags);                   \
        fprintf(output,                                                                  \
                "function _init_guru(ptr::Ptr{_nfst_plan{C%s}}, d::Integer,\n"           \
                "                    N::Vector{Cint}, M::Integer, n::Vector{Cint},\n"    \
                "                    m::Integer, flags::Integer, fftw_flags::Integer)\n" \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfst%s_init_guru, %s), Cvoid,\n"                            \
                "          (Ptr{_nfst_plan{C%s}}, Cint, Ptr{Cint}, Cint, Ptr{Cint},\n"   \
                "           Cint, Cuint, Cuint), ptr, d, N, M, n, m, flags,\n"           \
                "          fftw_flags)\n"                                                \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _finalize(ptr::Ptr{_nfst_plan{C%s}})\n"                        \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfst%s_finalize, %s), Cvoid,\n"                             \
                "          (Ptr{_nfst_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _precompute_psi(ptr::Ptr{_nfst_plan{C%s}})\n"                  \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfst%s_precompute_psi, %s), Cvoid,\n"                       \
                "          (Ptr{_nfst_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _apply_direct(ptr::Ptr{_nfst_plan{C%s}})\n"                    \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfst%s_trafo, %s), Cvoid,\n"                                \
                "          (Ptr{_nfst_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
        fprintf(output,                                                                  \
                "function _apply_adjoint(ptr::Ptr{_nfst_plan{C%s}})\n"                   \
                "    ptr != C_NULL || throw(NullPointerError())\n"                       \
                "    ccall((:nfst%s_adjoint, %s), Cvoid,\n"                              \
                "          (Ptr{_nfst_plan{C%s}},), ptr)\n"                              \
                "end\n", #real, SUFFIX(real), LIBRARY(real), #real);                     \
    } while (0)
    METHODS(nfstf_plan, float);
    METHODS(nfst_plan, double);
#undef METHODS

    return 0;
}
