package com.lelloman.simpleai.nlu

import android.util.Log
import java.io.InputStream
import java.io.DataInputStream
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
     * @param modelBuffer The model ByteBuffer (will be modified in place)
     * @param patchStream InputStream for the .lorapatch file
     * @param adapterId ID of the adapter being applied
     * @param adapterVersion Version of the adapter being applied
     * @return RevertPatch that can restore the model to its pre-patch state
     */
    fun applyPatch(
        modelBuffer: ByteBuffer,
        patchStream: InputStream,
        adapterId: String,
        adapterVersion: String
    ): RevertPatch {
        return applyPatch(modelBuffer, parsePatch(patchStream, modelBuffer.limit()), adapterId, adapterVersion)
    }

    fun applyPatch(modelBuffer: ByteBuffer, patches: List<Patch>, adapterId: String, adapterVersion: String): RevertPatch {
        validate(patches, modelBuffer.limit())
        Log.i(TAG, "Applying ${patches.size} patches for $adapterId v$adapterVersion")

        // Capture original bytes before patching (this becomes the revert patch)
        val revertPatches = patches.map { patch ->
            val originalBytes = ByteArray(patch.data.size)
            modelBuffer.position(patch.offset.toInt())
            modelBuffer.get(originalBytes)
            Patch(patch.offset, originalBytes)
        }

        // Apply patches
        for (patch in patches) {
            modelBuffer.position(patch.offset.toInt())
            modelBuffer.put(patch.data)
        }

        Log.i(TAG, "Applied ${patches.size} patches, revert patch is ${revertPatches.sumOf { it.data.size } / 1024} KB")
        return RevertPatch(adapterId, adapterVersion, revertPatches)
    }

    /**
     * Revert a previously applied patch, restoring the model to pristine state.
     *
     * @param modelBuffer The model ByteBuffer (will be modified in place)
     * @param revertPatch The revert patch from a previous applyPatch call
     */
    fun revertPatch(modelBuffer: ByteBuffer, revertPatch: RevertPatch) {
        Log.i(TAG, "Reverting ${revertPatch.patches.size} patches for ${revertPatch.adapterId}")

        for (patch in revertPatch.patches) {
            modelBuffer.position(patch.offset.toInt())
            modelBuffer.put(patch.data)
        }

        Log.i(TAG, "Reverted to pristine state")
    }

    /**
     * Parse a .lorapatch file.
     */
    internal fun parsePatch(inputStream: InputStream, modelSize: Int): List<Patch> {
        val input = DataInputStream(inputStream)
        val magic = ByteArray(4).also { input.readFully(it) }
        require(magic.contentEquals(MAGIC)) { "Invalid patch magic" }
        require(Integer.reverseBytes(input.readInt()) == VERSION) { "Unsupported patch version" }
        val count = Integer.reverseBytes(input.readInt())
        require(count in 0..4096) { "Patch count exceeds limit" }
        var total = 0L
        val patches = List(count) {
            val offset = java.lang.Long.reverseBytes(input.readLong())
            val length = Integer.reverseBytes(input.readInt())
            require(length in 1..(16 * 1024 * 1024)) { "Patch length exceeds limit" }
            require(offset >= 0 && offset <= modelSize.toLong() - length) { "Patch outside model bounds" }
            total += length
            require(total <= 128L * 1024 * 1024) { "Total patch size exceeds limit" }
            Patch(offset, ByteArray(length).also { input.readFully(it) })
        }
        require(input.read() == -1) { "Unexpected trailing patch data" }
        validate(patches, modelSize)
        return patches
    }

    private fun validate(patches: List<Patch>, modelSize: Int) {
        var end = 0L
        for (patch in patches.sortedBy { it.offset }) {
            require(patch.offset >= end && patch.offset <= modelSize.toLong() - patch.data.size) { "Overlapping or out-of-bounds patches" }
            end = patch.offset + patch.data.size
        }
    }
}
