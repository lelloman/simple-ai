# Android 16 KB validation

Ticket: [LLPR/AI-26](https://crumbles.lelloman.com/w/LLPR/AI/26)

Validated on 2026-09-14 after upgrading ONNX Runtime Android from 1.16.3 to
1.29.0. Other native inputs: llamacpp-kotlin 0.2.0, ML Kit translation 17.0.3,
language ID 17.0.6, and the tokenizers 0.22.2 JNI bridge built with NDK
27.0.12077973 and `-Wl,-z,max-page-size=16384`.

From `android/`:

```sh
./gradlew assembleDebug testDebugUnitTest lintDebug
python3 scripts/check-native-alignment.py app/build/outputs/apk/debug/app-debug.apk
```

All commands passed. The checker inspected all 12 packaged ARM64 shared
libraries, every ELF LOAD segment, and the ZIP offset of each uncompressed
library. Lint no longer reports the ONNX alignment warning. The check skips
32-bit ELF files because Android's 16 KB page-size requirement covers 64-bit
ABIs.

Runtime verification is **pending**: `adb devices -l` returned no devices;
the host has no `/dev/kvm`, and installed system images are x86/x86_64 while
the current llama dependency only ships ARM64 binaries. These static checks
do not establish successful inference on a 16 KB device.

To finish verification, use an ARM64 Android device booted with 16 KB pages,
confirm `adb shell getconf PAGE_SIZE` returns `16384`, install the complete
APK, and exercise NLU model loading/classification, local generation, and
translation. Record device/build, results, and relevant crash logs here.

References: [Android page-size guidance](https://developer.android.com/guide/practices/page-sizes),
[ONNX Runtime 1.29.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0).
