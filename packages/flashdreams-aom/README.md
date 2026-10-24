# FlashDreams AVIF tools

This required FlashDreams dependency builds upstream `avifenc` and `avifdec`
with libavif and libaom and installs the executables inside a platform wheel.
It uses no FFmpeg, Python bindings, system codec libraries, CUDA, or runtime
compilation. The libavif command-line tools also link their PNG/JPEG I/O
dependencies: zlib, libpng, and libjpeg-turbo.

Normal syncs install this dependency automatically, building both executables
with PEP 517 isolation when needed. From the repository root:

```sh
uv sync
```

Source builds require Git and a C/C++ compiler. Windows requires Visual Studio
Build Tools with Desktop development with C++ and a Windows SDK; run source
installs from an x64 Visual Studio Developer PowerShell. Install YASM
on x86 Linux/macOS; Windows downloads it automatically. PEP 517 isolation
provides scikit-build-core, CMake, and Ninja.

Raw frames are piped into the encoder. libavif retains compressed samples until
the file is finalized, so encoder memory grows with recording size. Growing
the canvas decodes prior frames to temporary Y4M files and re-encodes them;
this needs temporary disk space and can introduce additional quality loss.
