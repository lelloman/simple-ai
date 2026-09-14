# UI metadata and diagnostics

- `translation/Language.kt` owns all 59 supported language codes and ML Kit mappings. Display/native names come from Java/Android locale data. The translation manager and all UI lists use this catalog.
- `res/values/strings.xml` owns screen text, accessible action names, setup copy, and connection/deletion feedback. English is the current translation; add matching keys under `values-<locale>` for additional translations. API/native exception messages are diagnostic strings and may remain English.
- `model/NluModel.kt` and `model/AvailableModels.kt` own pinned artifact identity, revision, SHA-256 and exact byte count. Capability sizes and dialogs reference them. `ui/ByteSize.kt` formats decimal kB/MB/GB consistently; ML Kit pack size is explicitly approximate.
- About provides support/model/license links and copies a diagnostic report containing build/protocol, Android API/ABIs, model revisions/hashes/sizes and inference-library versions. The report excludes prompts, output, auth tokens, client package approvals and cloud endpoint configuration.

Model license sources: [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0), [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) (MIT), and [ML Kit terms/privacy](https://developers.google.com/ml-kit/terms). About links directly to these sources.

The catalog and exact byte/unit consistency have JVM regression tests. Resource format arguments are checked by Android lint. Translation quality and localized layout review are required when adding a locale; moving English copy into resources does not by itself add translated UI.
