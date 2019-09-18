#
# Define the following environment variables if the default definitions is not
# suitable for your installation.
#
#     NFFT_DEFS specifies the C-preprocessor flags for finding headers "nfft3.h".
#              For instance: NFFT_DEFS=-I/usr/local/include
#
#     NFFT_LIBS specifies the linker flags for linking with the NFFT dynamic
#              library. For instance: NFFT_LIBS="-L/usr/local/lib -lnfft3 -lfftw3"
#
#     LIB_NFFT_DLL specifies the path to the NFFT dynamic library (double
#              precision version).  This may be defined to force using another
#              dynamic library than the one imposed by the system.  Note that,
#              this variable has precedence over NFFT_LIBS.  For instance:
#              NFFT_DLL=/usr/local/libnfft3.so
#
#     LIB_NFFTF_DLL specifies the path to the NFFT dynamic library (single precision version).
#

srcdir = .

NFFT_DEFS=
NFFT_LIBS=-lnfft3 -lfftw3
LIB_NFFT_DLL=libnfft3.so
LIB_NFFTF_DLL=libnfft3f.so

all: deps.jl

clean:
	rm -f *~ core gendeps

distclean: clean
	rm -f deps.jl

# FIXME: compile a simple test executable as part of the installation.

gendeps: ${srcdir}/gendeps.c Makefile
	$(CC) $(CPPFLAGS) $(NFFT_DEFS) \
	    -DLIB_NFFTF_DLL="\"$(LIB_NFFTF_DLL)\"" \
	    -DLIB_NFFT_DLL="\"$(LIB_NFFT_DLL)\"" \
	    $(CFLAGS) -o "$@" "$<" $(NFFT_LIBS) $(LDFLAGS)

deps.jl: gendeps Makefile
	rm -f "$@"
	./gendeps >"$@"
	chmod 444 "$@"

.PHONY: all clean distclean
