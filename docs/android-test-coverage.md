# Android checks

Run `cd android && ./scripts/check` with JDK 17, Android SDK 36, NDK 27.0.12077973, Rust 1.96.0, cargo-ndk 4.1.2 and Android arm64/armv7 Rust targets installed. `./scripts/check --offline` uses cached Gradle dependencies (Cargo also needs a populated cache).

The dedicated Android GitHub workflow runs on Android changes in pull requests and pushes to main/master, plus manual dispatch. It builds the debug APK, runs JVM tests and lint, compiles the instrumentation APK, runs Rust tokenizer tests, and verifies every packaged 64-bit native library's ELF/ZIP 16 KB alignment. Test/lint reports and the instrumentation APK are retained even when a check fails. It does not publish an app or use signing/cloud secrets.

Regression coverage includes:

| Concern | Tests |
|---|---|
| First-run translation setup, large text/narrow card actions | CapabilityAccessibilityTest (instrumentation) |
| Bound service parcel contract, current/old protocol and invalid generation | ServiceContractTest (instrumentation) |
| UI/service protocol and translation parsing | ServiceInfoClientTest, ServiceTranslationClientTest |
| Activation/deletion/lazy reload and idle lifetime | ManagedModelTest, IdleResourcesTest |
| Interrupted downloads, response cleanup and range/identity integrity | ResumableDownloadTest, CancellableCallTest, DownloadPolicyTest |
| Generation timeout outcomes, concurrent lifetime and event ordering | LlamaEngineTest |
| Request deadlines, caller cancellation, schema and nullable cloud usage | RequestContractTest, CallerBudgetTest |
| Adapter selection, binary bounds, resource closure and restart integrity | AdapterAccessTest, AdapterFilesTest, WorkingModelFileTest, NativeResourcesTest |
| Multilingual tokenizer parity | NativeTokenizerTest and tokenizer-native Rust tests |
| Translation draft/language concurrency and metadata | TranslationSessionTest, KeyedDownloadsTest, LanguageSearchTest, MetadataTest |

Run `./gradlew connectedDebugAndroidTest` on a suitable connected device for instrumentation execution. CI compiles those tests but does not claim to run ARM64 inference on an x86 runner. Parcel tests force the generated AIDL proxy in the target process; independently signed client approval tests still need a second app. A real 16 KB ARM64 device/emulator remains necessary for native runtime acceptance. See the accessibility and 16 KB checklists.

The workflow has been validated through its local check command; a hosted run can only be observed after the branch is pushed. No hosted CI result is claimed for unpushed commits.
