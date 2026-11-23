#!/bin/bash
# On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).
#
# Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng
#
# Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
#
# Codec pruning is a contribution of this paper; please cite it if you use, modify, or port this code.
#
# Builds the codec reader of the codec gazer: FFmpeg 5.1 (H.264 decoder only) with
# src/codecopsd/codec_gazer/csrc/ffmpeg-5.1.patch, linked statically into src/codecopsd/codec_gazer/_codec_reader.abi3.so
# (CPython >= 3.10). Needs curl, bzip2, patch, a C compiler, make and the Python development headers (Python.h) of
# $PYTHON (default python3).
#
#     bash scripts/build_codec_reader.sh [build dir]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$(mkdir -p "${1:-$ROOT/build/ffmpeg}" && cd "${1:-$ROOT/build/ffmpeg}" && pwd)
INCLUDE=$("${PYTHON:-python3}" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
[ -f "$INCLUDE/Python.h" ] || { echo "Python.h not found in $INCLUDE: install the Python development headers" >&2; exit 1; }
cd "$BUILD"

[ -f ffmpeg-5.1.tar.bz2 ] || curl -fsSLo ffmpeg-5.1.tar.bz2 https://ffmpeg.org/releases/ffmpeg-5.1.tar.bz2
echo "32b56fb01ce90d452958ae25e91c9564abf49ed5453c127bec23c63e530aa8fa  ffmpeg-5.1.tar.bz2" | sha256sum -c --quiet
rm -rf ffmpeg-5.1 && tar xjf ffmpeg-5.1.tar.bz2
cd ffmpeg-5.1
patch -p1 < "$ROOT/src/codecopsd/codec_gazer/csrc/ffmpeg-5.1.patch"
# No assembly: the SIMD IDCTs would replace the patched C ones. The h263 decoder is only there because FFmpeg 5.1
# exports motion vectors only when it is built with a decoder of the mpegvideo family.
./configure --cc="${CC:-cc}" --prefix="$BUILD/install" --enable-pic --disable-asm --disable-autodetect \
    --disable-everything --disable-programs --disable-doc --disable-avdevice --disable-avfilter --disable-swscale \
    --disable-swresample --disable-network --enable-decoder=h264 --enable-decoder=h263 --enable-demuxer=mov \
    --enable-parser=h264 --enable-protocol=file > configure.log 2>&1 || { tail -n 20 configure.log >&2; exit 1; }
make -j"$(nproc)" > make.log 2>&1 && make install > /dev/null || { tail -n 20 make.log >&2; exit 1; }

"${CC:-cc}" -O2 -fPIC -shared -I"$INCLUDE" -I"$BUILD/install/include" "$ROOT/src/codecopsd/codec_gazer/csrc/codec_reader.c" \
    -o "$ROOT/src/codecopsd/codec_gazer/_codec_reader.abi3.so" -L"$BUILD/install/lib" -lavformat -lavcodec -lavutil \
    -lm -latomic -pthread -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
echo "built $ROOT/src/codecopsd/codec_gazer/_codec_reader.abi3.so"
