# A Julia interface to the NFFT library

| **Documentation**               | **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![][doc-dev-img]][doc-dev-url] | [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

The LibNFFT package provides a Julia interface to the [NFFT library][nfft-url]
which provides fast nonequispaced Fourier transforms and variants.

Currently the following transforms are interfaced:

- **NFFT** the nonequispaced fast Fourier transform;
- **NFCT** the nonequispaced fast cosine transform;
- **NFST** the nonequispaced fast sine transform;


## Installation

LibNFFT can be installed by Julia's package manager:

```julia
pkg> add https://github.com/emmt/LibNFFT.jl
```

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/LibNFFT.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/LibNFFT.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.com/emmt/LibNFFT.jl.svg?branch=master
[travis-url]: https://travis-ci.com/emmt/LibNFFT.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/LibNFFT.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/LibNFFT-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/LibNFFT.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/LibNFFT.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/LibNFFT.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/LibNFFT.jl?branch=master

[nfft-url]: http://www-user.tu-chemnitz.de/~potts/nfft/
