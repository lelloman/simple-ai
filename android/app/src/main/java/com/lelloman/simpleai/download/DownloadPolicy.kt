package com.lelloman.simpleai.download

import android.content.Context
import java.io.File

object DownloadPolicy {
    const val RESERVE_BYTES = 64L * 1024 * 1024
    fun allowsMetered(context: Context) = context.getSharedPreferences("downloads", Context.MODE_PRIVATE).getBoolean("metered", false)
    fun setAllowsMetered(context: Context, allowed: Boolean) {
        context.getSharedPreferences("downloads", Context.MODE_PRIVATE).edit().putBoolean("metered", allowed).apply()
    }
    fun requiredBytes(size: Long, partial: Long, workingCopy: Long = 0): Long =
        (size - partial.coerceIn(0, size)).coerceAtLeast(0) + workingCopy + RESERVE_BYTES

    fun usedBytes(directory: File): Long = directory.walkTopDown().filter { it.isFile }.sumOf { it.length() }
}
