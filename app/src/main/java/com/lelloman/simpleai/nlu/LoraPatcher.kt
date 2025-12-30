package com.lelloman.simpleai.nlu

import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Applies LoRA patch files to ONNX models in memory.
 *
 * Maintains the ability to revert patches so a single in-memory model
 * can be switched between different adapters without reloading from disk.
 *
 * Patch file format (.lorapatch):
 * - 4 bytes: magic "LORA"
 * - 4 bytes: version (uint32, little-endian) = 1
 * - 4 bytes: number of patches (uint32)
 * - For each patch:
 *   - 8 bytes: offset in ONNX file (uint64)
 *   - 4 bytes: data length in bytes (uint32)
 *   - N bytes: new weight data
 */
class LoraPatcher {

    companion object {
        private const val TAG = "LoraPatcher"
        private val MAGIC = byteArrayOf('L'.code.toByte(), 'O'.code.toByte(), 'R'.code.toByte(), 'A'.code.toByte())
        private const val VERSION = 1
    }

    data class Patch(
        val offset: Long,
        val data: ByteArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Patch) return false
            return offset == other.offset && data.contentEquals(other.data)
        }

        override fun hashCode(): Int = 31 * offset.hashCode() + data.contentHashCode()
    }

    /**
     * A revert patch that can restore the model to its previous state.
     */
    data class RevertPatch(
        val adapterId: String,
        val adapterVersion: String,
        val patches: List<Patch>
    )

    /**
     * Apply a LoRA patch to an in-memory model, returning a revert patch.
     *
     * @param modelBytes The model bytes (will be modified in place)
     * @param patchStream InputStream for the .lorapatch file
     * @param adapterId ID of the adapter being applied
     * @param adapterVersion Version of the adapter being applied
     * @return RevertPatch that can restore the model to its pre-patch state
     */
    fun applyPatch(
        modelBytes: ByteArray,
        patchStream: InputStream,
        adapterId: String,
        adapterVersion: String
    ): RevertPatch {
        val patches = parsePatch(patchStream)
        Log.i(TAG, "Applying ${patches.size} patches for $adapterId v$adapterVersion")

        // Capture original bytes before patching (this becomes the revert patch)
        val revertPatches = patches.map { patch ->
            val originalBytes = ByteArray(patch.data.size)
            System.arraycopy(modelBytes, patch.offset.toInt(), originalBytes, 0, patch.data.size)
            Patch(patch.offset, originalBytes)
        }

        // Apply patches
        for (patch in patches) {
            System.arraycopy(patch.data, 0, modelBytes, patch.offset.toInt(), patch.data.size)
        }

        Log.i(TAG, "Applied ${patches.size} patches, revert patch is ${revertPatches.sumOf { it.data.size } / 1024} KB")
        return RevertPatch(adapterId, adapterVersion, revertPatches)
    }

    /**
     * Revert a previously applied patch, restoring the model to pristine state.
     *
     * @param modelBytes The model bytes (will be modified in place)
     * @param revertPatch The revert patch from a previous applyPatch call
     */
    fun revertPatch(modelBytes: ByteArray, revertPatch: RevertPatch) {
        Log.i(TAG, "Reverting ${revertPatch.patches.size} patches for ${revertPatch.adapterId}")

        for (patch in revertPatch.patches) {
            System.arraycopy(patch.data, 0, modelBytes, patch.offset.toInt(), patch.data.size)
        }

        Log.i(TAG, "Reverted to pristine state")
    }

    /**
     * Parse a .lorapatch file.
     */
    private fun parsePatch(inputStream: InputStream): List<Patch> {
        val patches = mutableListOf<Patch>()

        // Read header
        val magic = ByteArray(4)
        inputStream.read(magic)
        require(magic.contentEquals(MAGIC)) { "Invalid patch file magic: ${magic.contentToString()}" }

        val headerBuffer = ByteArray(8)
        inputStream.read(headerBuffer)
        val header = ByteBuffer.wrap(headerBuffer).order(ByteOrder.LITTLE_ENDIAN)
        val version = header.int
        val numPatches = header.int

        require(version == VERSION) { "Unsupported patch version: $version" }
        Log.i(TAG, "Patch file version $version with $numPatches patches")

        // Read patches
        for (i in 0 until numPatches) {
            val patchHeader = ByteArray(12)
            inputStream.read(patchHeader)
            val patchBuf = ByteBuffer.wrap(patchHeader).order(ByteOrder.LITTLE_ENDIAN)
            val offset = patchBuf.long
            val dataLength = patchBuf.int

            val data = ByteArray(dataLength)
            var bytesRead = 0
            while (bytesRead < dataLength) {
                val read = inputStream.read(data, bytesRead, dataLength - bytesRead)
                if (read == -1) break
                bytesRead += read
            }
            require(bytesRead == dataLength) { "Incomplete patch data" }

            patches.add(Patch(offset, data))
        }

        return patches
    }
}
