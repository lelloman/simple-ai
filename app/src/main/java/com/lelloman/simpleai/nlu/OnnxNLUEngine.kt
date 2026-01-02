package com.lelloman.simpleai.nlu

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.ParcelFileDescriptor
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.*
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.LongBuffer
import java.util.concurrent.TimeUnit

/**
 * NLU Engine using ONNX Runtime with in-memory LoRA patching.
 *
 * Architecture:
 * 1. Base XLM-RoBERTa model is downloaded once and kept in memory
 * 2. Adapters are provided by caller apps via ParcelFileDescriptor
 * 3. When switching adapters: revert current patch → apply new patch → recreate session
 * 4. Classification heads are applied after encoder inference
 */
class OnnxNLUEngine(
    private val context: Context
) : NLUEngine {

    companion object {
        private const val TAG = "OnnxNLUEngine"
        private const val CLS_TOKEN_ID = 0L
        private const val SEP_TOKEN_ID = 2L
        private const val PAD_TOKEN_ID = 1L
        private const val UNK_TOKEN_ID = 3L

        private const val BASE_MODEL_URL = "https://huggingface.co/lelloman/xlm-roberta-base-onnx-int8/resolve/main/xlm_roberta_base_int8.onnx"
        private const val BASE_MODEL_FILE = "xlm_roberta_base_int8.onnx"
    }

    sealed class Status {
        data object NotInitialized : Status()
        data class Downloading(val progress: Float, val message: String) : Status()
        data object Ready : Status()
        data object Patching : Status()
        data class Error(val message: String) : Status()
    }

    private val _status = MutableStateFlow<Status>(Status.NotInitialized)
    val status: StateFlow<Status> = _status

    private var ortEnv: OrtEnvironment? = null
    private var modelBytes: ByteArray? = null
    private var currentSession: OrtSession? = null
    private var currentRevertPatch: LoraPatcher.RevertPatch? = null
    private var currentAdapter: LoadedAdapter? = null

    private val mutex = Mutex()
    private val loraPatcher = LoraPatcher()

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private val nluDir: File
        get() = File(context.filesDir, "nlu_models")

    private val baseModelFile: File
        get() = File(nluDir, BASE_MODEL_FILE)

    data class LoadedAdapter(
        val id: String,
        val version: String,
        val intentHead: FloatArray,
        val intentBias: FloatArray,
        val slotHead: FloatArray,
        val slotBias: FloatArray,
        val numIntents: Int,
        val numSlots: Int,
        val vocab: Map<String, Long>,
        val merges: List<Pair<String, String>>,
        val maxLength: Int,
        val intents: List<String>,
        val slotLabels: List<String>
    )

    override val isReady: Boolean
        get() = modelBytes != null && _status.value == Status.Ready

    override val adapters: List<NLUAdapter>
        get() = currentAdapter?.let {
            listOf(NLUAdapter(
                id = it.id,
                name = it.id,
                version = it.version,
                baseModel = "xlm-roberta-base",
                maxLength = it.maxLength,
                intents = it.intents,
                slotLabels = it.slotLabels
            ))
        } ?: emptyList()

    override suspend fun initialize(): Result<Unit> = withContext(Dispatchers.IO) {
        mutex.withLock {
            try {
                Log.i(TAG, "Initializing ONNX NLU Engine...")
                ortEnv = OrtEnvironment.getEnvironment()
                nluDir.mkdirs()

                // Load base model into memory
                loadBaseModel()

                _status.value = Status.Ready
                Log.i(TAG, "NLU Engine initialized, model loaded (${modelBytes?.size?.div(1024 * 1024)} MB)")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize NLU Engine", e)
                _status.value = Status.Error(e.message ?: "Initialization failed")
                Result.failure(e)
            }
        }
    }

    private suspend fun loadBaseModel() {
        // Download if needed
        if (!baseModelFile.exists()) {
            downloadBaseModel()
        }

        // Load into memory
        Log.i(TAG, "Loading base model into memory...")
        modelBytes = baseModelFile.readBytes()
        Log.i(TAG, "Base model loaded: ${modelBytes?.size?.div(1024 * 1024)} MB")
    }

    private suspend fun downloadBaseModel() {
        Log.i(TAG, "Downloading base model...")
        _status.value = Status.Downloading(0f, "Downloading base model...")

        val request = Request.Builder().url(BASE_MODEL_URL).build()
        val response = httpClient.newCall(request).execute()

        if (!response.isSuccessful) {
            throw RuntimeException("Download failed: ${response.code} ${response.message}")
        }

        val body = response.body ?: throw RuntimeException("Empty response body")
        val contentLength = body.contentLength()

        body.byteStream().use { input ->
            FileOutputStream(baseModelFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var totalBytesRead = 0L

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead

                    if (contentLength > 0) {
                        val progress = totalBytesRead.toFloat() / contentLength
                        _status.value = Status.Downloading(progress, "Downloading... ${(progress * 100).toInt()}%")
                    }
                }
            }
        }

        Log.i(TAG, "Base model downloaded: ${baseModelFile.length() / 1024 / 1024} MB")
    }

    /**
     * Apply an adapter provided by the caller app.
     *
     * @param adapterId Unique identifier for this adapter
     * @param adapterVersion Version string for cache invalidation
     * @param patchFd ParcelFileDescriptor for the .lorapatch file
     * @param headsFd ParcelFileDescriptor for the heads.bin file
     * @param tokenizerFd ParcelFileDescriptor for the tokenizer.json file
     * @param configFd ParcelFileDescriptor for the config.json file
     */
    suspend fun applyAdapter(
        adapterId: String,
        adapterVersion: String,
        patchFd: ParcelFileDescriptor,
        headsFd: ParcelFileDescriptor,
        tokenizerFd: ParcelFileDescriptor,
        configFd: ParcelFileDescriptor
    ): Result<Unit> = withContext(Dispatchers.IO) {
        mutex.withLock {
            try {
                val bytes = modelBytes ?: return@withContext Result.failure(
                    IllegalStateException("Model not loaded")
                )

                // Check if already using this adapter
                if (currentAdapter?.id == adapterId && currentAdapter?.version == adapterVersion) {
                    Log.i(TAG, "Adapter $adapterId v$adapterVersion already applied")
                    return@withContext Result.success(Unit)
                }

                _status.value = Status.Patching

                // Revert current patch if any
                currentRevertPatch?.let { revert ->
                    Log.i(TAG, "Reverting current patch: ${revert.adapterId}")
                    loraPatcher.revertPatch(bytes, revert)
                    currentSession?.close()
                    currentSession = null
                    currentRevertPatch = null
                    currentAdapter = null
                }

                // Apply new patch
                Log.i(TAG, "Applying adapter $adapterId v$adapterVersion")
                FileInputStream(patchFd.fileDescriptor).use { patchStream ->
                    currentRevertPatch = loraPatcher.applyPatch(bytes, patchStream, adapterId, adapterVersion)
                }

                // Load heads
                val headsData = FileInputStream(headsFd.fileDescriptor).use { headsStream ->
                    loadHeads(headsStream)
                }

                // Load tokenizer
                val (vocab, merges) = FileInputStream(tokenizerFd.fileDescriptor).use { tokenizerStream ->
                    loadTokenizer(tokenizerStream)
                }

                // Load config
                val (intents, slotLabels, maxLength) = FileInputStream(configFd.fileDescriptor).use { configStream ->
                    loadConfig(configStream)
                }

                // Create new ONNX session from patched bytes
                val env = ortEnv ?: return@withContext Result.failure(
                    IllegalStateException("ORT environment not initialized")
                )
                val sessionOptions = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(4)
                }
                currentSession = env.createSession(bytes, sessionOptions)

                currentAdapter = LoadedAdapter(
                    id = adapterId,
                    version = adapterVersion,
                    intentHead = headsData.intentHead,
                    intentBias = headsData.intentBias,
                    slotHead = headsData.slotHead,
                    slotBias = headsData.slotBias,
                    numIntents = headsData.numIntents,
                    numSlots = headsData.numSlots,
                    vocab = vocab,
                    merges = merges,
                    maxLength = maxLength,
                    intents = intents,
                    slotLabels = slotLabels
                )

                _status.value = Status.Ready
                Log.i(TAG, "Adapter $adapterId v$adapterVersion applied successfully")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to apply adapter", e)
                _status.value = Status.Error(e.message ?: "Failed to apply adapter")
                Result.failure(e)
            }
        }
    }

    /**
     * Remove current adapter, reverting to pristine model state.
     */
    suspend fun removeAdapter(): Result<Unit> = withContext(Dispatchers.IO) {
        mutex.withLock {
            try {
                val bytes = modelBytes ?: return@withContext Result.failure(
                    IllegalStateException("Model not loaded")
                )

                currentRevertPatch?.let { revert ->
                    Log.i(TAG, "Reverting patch: ${revert.adapterId}")
                    loraPatcher.revertPatch(bytes, revert)
                }

                currentSession?.close()
                currentSession = null
                currentRevertPatch = null
                currentAdapter = null

                _status.value = Status.Ready
                Log.i(TAG, "Adapter removed, model is pristine")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to remove adapter", e)
                Result.failure(e)
            }
        }
    }

    private data class HeadsData(
        val intentHead: FloatArray,
        val intentBias: FloatArray,
        val slotHead: FloatArray,
        val slotBias: FloatArray,
        val numIntents: Int,
        val numSlots: Int
    )

    private fun loadHeads(inputStream: java.io.InputStream): HeadsData {
        val headerBuffer = ByteArray(8)
        val sizeBuffer = ByteArray(4)

        // Intent head weight
        inputStream.read(headerBuffer)
        val intentHeader = ByteBuffer.wrap(headerBuffer).order(ByteOrder.LITTLE_ENDIAN)
        val intentRows = intentHeader.int
        val intentCols = intentHeader.int
        val intentSize = intentRows * intentCols

        val intentBytes = ByteArray(intentSize * 4)
        inputStream.read(intentBytes)
        val intentHead = FloatArray(intentSize)
        ByteBuffer.wrap(intentBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(intentHead)

        // Slot head weight
        inputStream.read(headerBuffer)
        val slotHeader = ByteBuffer.wrap(headerBuffer).order(ByteOrder.LITTLE_ENDIAN)
        val slotRows = slotHeader.int
        val slotCols = slotHeader.int
        val slotSize = slotRows * slotCols

        val slotBytes = ByteArray(slotSize * 4)
        inputStream.read(slotBytes)
        val slotHead = FloatArray(slotSize)
        ByteBuffer.wrap(slotBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(slotHead)

        // Intent bias
        inputStream.read(sizeBuffer)
        val intentBiasSize = ByteBuffer.wrap(sizeBuffer).order(ByteOrder.LITTLE_ENDIAN).int
        val intentBiasBytes = ByteArray(intentBiasSize * 4)
        inputStream.read(intentBiasBytes)
        val intentBias = FloatArray(intentBiasSize)
        ByteBuffer.wrap(intentBiasBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(intentBias)

        // Slot bias
        inputStream.read(sizeBuffer)
        val slotBiasSize = ByteBuffer.wrap(sizeBuffer).order(ByteOrder.LITTLE_ENDIAN).int
        val slotBiasBytes = ByteArray(slotBiasSize * 4)
        inputStream.read(slotBiasBytes)
        val slotBias = FloatArray(slotBiasSize)
        ByteBuffer.wrap(slotBiasBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(slotBias)

        Log.i(TAG, "Loaded heads: intent=${intentRows}x${intentCols} + bias[$intentBiasSize], slot=${slotRows}x${slotCols} + bias[$slotBiasSize]")
        return HeadsData(intentHead, intentBias, slotHead, slotBias, intentRows, slotRows)
    }

    private fun loadTokenizer(inputStream: java.io.InputStream): Pair<Map<String, Long>, List<Pair<String, String>>> {
        val tokenizerJson = inputStream.bufferedReader().readText()
        val tokenizer = Json.parseToJsonElement(tokenizerJson).jsonObject

        val model = tokenizer["model"]!!.jsonObject
        val vocabArray = model["vocab"]!!.jsonObject

        val vocab = vocabArray.entries.associate { (token, id) ->
            token to id.jsonPrimitive.long
        }

        val mergesArray = model["merges"]?.jsonArray ?: JsonArray(emptyList())
        val merges = mergesArray.map { mergeElement ->
            val parts = mergeElement.jsonPrimitive.content.split(" ")
            parts[0] to parts[1]
        }

        Log.i(TAG, "Loaded tokenizer: ${vocab.size} vocab, ${merges.size} merges")
        return vocab to merges
    }

    private data class ConfigData(
        val intents: List<String>,
        val slotLabels: List<String>,
        val maxLength: Int
    )

    private fun loadConfig(inputStream: java.io.InputStream): ConfigData {
        val configJson = inputStream.bufferedReader().readText()
        val config = Json.parseToJsonElement(configJson).jsonObject

        val intents = config["intents"]!!.jsonArray.map { it.jsonPrimitive.content }
        val slotLabels = config["slot_labels"]!!.jsonArray.map { it.jsonPrimitive.content }
        val maxLength = config["max_length"]?.jsonPrimitive?.int ?: 64

        Log.i(TAG, "Loaded config: ${intents.size} intents, ${slotLabels.size} slots, maxLength=$maxLength")
        return ConfigData(intents, slotLabels, maxLength)
    }

    override suspend fun classify(text: String, adapterId: String): Result<ClassificationResult> =
        withContext(Dispatchers.IO) {
            try {
                val adapter = currentAdapter
                val session = currentSession

                if (adapter == null || session == null) {
                    return@withContext Result.failure(
                        IllegalStateException("No adapter loaded. Call applyAdapter first.")
                    )
                }

                if (adapter.id != adapterId) {
                    return@withContext Result.failure(
                        IllegalArgumentException("Adapter mismatch: loaded=${adapter.id}, requested=$adapterId")
                    )
                }

                // Tokenize
                val (inputIds, attentionMask) = tokenize(text, adapter.vocab, adapter.merges, adapter.maxLength)

                // Run encoder inference
                val env = ortEnv ?: return@withContext Result.failure(IllegalStateException("ORT not initialized"))

                val inputIdsTensor = OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(inputIds),
                    longArrayOf(1, inputIds.size.toLong())
                )
                val attentionMaskTensor = OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(attentionMask),
                    longArrayOf(1, attentionMask.size.toLong())
                )

                val inputs = mapOf(
                    "input_ids" to inputIdsTensor,
                    "attention_mask" to attentionMaskTensor
                )

                val outputs = session.run(inputs)

                // Get encoder output (last_hidden_state)
                @Suppress("UNCHECKED_CAST")
                val hiddenStates = outputs[0].value as Array<Array<FloatArray>>
                val sequenceOutput = hiddenStates[0]
                val pooledOutput = sequenceOutput[0]

                // Apply classification heads
                val intentLogits = applyLinear(pooledOutput, adapter.intentHead, adapter.intentBias, adapter.numIntents)
                val slotLogits = sequenceOutput.map { tokenHidden ->
                    applyLinear(tokenHidden, adapter.slotHead, adapter.slotBias, adapter.numSlots)
                }

                // Get intent
                val intentIdx = intentLogits.indices.maxByOrNull { intentLogits[it] } ?: 0
                val intent = adapter.intents[intentIdx]
                val intentConfidence = softmax(intentLogits)[intentIdx]

                // Get slot labels
                val rawSlotLabels = slotLogits.map { tokenLogits ->
                    val slotIdx = tokenLogits.indices.maxByOrNull { tokenLogits[it] } ?: 0
                    adapter.slotLabels[slotIdx]
                }

                // Extract slots from BIO tags
                val slots = extractSlots(inputIds, rawSlotLabels, adapter.vocab)

                // Clean up
                inputIdsTensor.close()
                attentionMaskTensor.close()
                outputs.close()

                Result.success(
                    ClassificationResult(
                        intent = intent,
                        intentConfidence = intentConfidence,
                        slots = slots,
                        rawSlotLabels = rawSlotLabels
                    )
                )
            } catch (e: Exception) {
                Log.e(TAG, "Classification failed", e)
                Result.failure(e)
            }
        }

    private fun applyLinear(input: FloatArray, weights: FloatArray, bias: FloatArray, outFeatures: Int): FloatArray {
        val inFeatures = input.size
        val output = FloatArray(outFeatures)
        for (i in 0 until outFeatures) {
            var sum = bias[i]
            for (j in 0 until inFeatures) {
                sum += input[j] * weights[i * inFeatures + j]
            }
            output[i] = sum
        }
        return output
    }

    private fun tokenize(
        text: String,
        vocab: Map<String, Long>,
        merges: List<Pair<String, String>>,
        maxLength: Int
    ): Pair<LongArray, LongArray> {
        val tokens = bpeTokenize(text.lowercase(), vocab, merges)

        val inputIds = LongArray(maxLength) { PAD_TOKEN_ID }
        val attentionMask = LongArray(maxLength) { 0L }

        inputIds[0] = CLS_TOKEN_ID
        attentionMask[0] = 1L

        val maxTokens = minOf(tokens.size, maxLength - 2)
        for (i in 0 until maxTokens) {
            inputIds[i + 1] = tokens[i]
            attentionMask[i + 1] = 1L
        }

        inputIds[maxTokens + 1] = SEP_TOKEN_ID
        attentionMask[maxTokens + 1] = 1L

        return inputIds to attentionMask
    }

    private fun bpeTokenize(text: String, vocab: Map<String, Long>, merges: List<Pair<String, String>>): List<Long> {
        val result = mutableListOf<Long>()
        val words = text.split(Regex("\\s+")).filter { it.isNotEmpty() }

        for ((wordIdx, word) in words.withIndex()) {
            val processedWord = if (wordIdx == 0) word else "▁$word"
            var tokens = processedWord.map { it.toString() }.toMutableList()

            for ((first, second) in merges) {
                var i = 0
                while (i < tokens.size - 1) {
                    if (tokens[i] == first && tokens[i + 1] == second) {
                        tokens[i] = first + second
                        tokens.removeAt(i + 1)
                    } else {
                        i++
                    }
                }
            }

            for (token in tokens) {
                val id = vocab[token] ?: vocab["▁$token"] ?: UNK_TOKEN_ID
                result.add(id)
            }
        }

        return result
    }

    private fun extractSlots(
        inputIds: LongArray,
        slotLabels: List<String>,
        vocab: Map<String, Long>
    ): Map<String, List<String>> {
        val slots = mutableMapOf<String, MutableList<String>>()
        val reverseVocab = vocab.entries.associate { it.value to it.key }

        var currentSlot: String? = null
        var currentValue = StringBuilder()

        for (i in 1 until slotLabels.size) {
            val label = slotLabels[i]
            val tokenId = inputIds[i]

            if (tokenId == SEP_TOKEN_ID || tokenId == PAD_TOKEN_ID) break

            val token = reverseVocab[tokenId] ?: continue

            when {
                label.startsWith("B-") -> {
                    if (currentSlot != null && currentValue.isNotEmpty()) {
                        val cleanValue = currentValue.toString().replace("▁", " ").trim()
                        if (cleanValue.isNotEmpty()) {
                            slots.getOrPut(currentSlot) { mutableListOf() }.add(cleanValue)
                        }
                    }
                    currentSlot = label.removePrefix("B-")
                    currentValue = StringBuilder(token)
                }
                label.startsWith("I-") && currentSlot == label.removePrefix("I-") -> {
                    currentValue.append(token)
                }
                else -> {
                    if (currentSlot != null && currentValue.isNotEmpty()) {
                        val cleanValue = currentValue.toString().replace("▁", " ").trim()
                        if (cleanValue.isNotEmpty()) {
                            slots.getOrPut(currentSlot) { mutableListOf() }.add(cleanValue)
                        }
                    }
                    currentSlot = null
                    currentValue = StringBuilder()
                }
            }
        }

        if (currentSlot != null && currentValue.isNotEmpty()) {
            val cleanValue = currentValue.toString().replace("▁", " ").trim()
            if (cleanValue.isNotEmpty()) {
                slots.getOrPut(currentSlot) { mutableListOf() }.add(cleanValue)
            }
        }

        return slots
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { kotlin.math.exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { (it / sumExps).toFloat() }.toFloatArray()
    }

    override fun release() {
        currentSession?.close()
        currentSession = null
        currentRevertPatch = null
        currentAdapter = null
        modelBytes = null
        ortEnv?.close()
        ortEnv = null
    }
}
