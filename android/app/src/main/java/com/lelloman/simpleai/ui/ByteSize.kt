package com.lelloman.simpleai.ui

import java.util.Locale

/** Decimal units everywhere: a MB is 1,000,000 bytes. */
internal fun formatSize(bytes: Long): String = when {
    bytes >= 1_000_000_000 -> String.format(Locale.getDefault(), "%.2f GB", bytes / 1_000_000_000.0)
    bytes >= 1_000_000 -> String.format(Locale.getDefault(), "%.1f MB", bytes / 1_000_000.0)
    bytes >= 1_000 -> String.format(Locale.getDefault(), "%.1f kB", bytes / 1_000.0)
    else -> "$bytes B"
}
