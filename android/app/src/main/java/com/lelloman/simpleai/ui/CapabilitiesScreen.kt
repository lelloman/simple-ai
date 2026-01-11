package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.lelloman.simpleai.capability.CapabilityStatus

private enum class DeleteConfirmation {
    VOICE_COMMANDS,
    LOCAL_AI
}

/**
 * Main screen showing all capabilities as cards.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CapabilitiesScreen(
    viewModel: CapabilitiesViewModel = viewModel(),
    onNavigateToTranslationLanguages: () -> Unit = {},
    onNavigateToTranslationTest: () -> Unit = {},
    onNavigateToAbout: () -> Unit = {}
) {
    val state by viewModel.state.collectAsState()
    var deleteConfirmation by remember { mutableStateOf<DeleteConfirmation?>(null) }

    // Delete confirmation dialog
    deleteConfirmation?.let { confirmation ->
        val (title, size, onConfirm) = when (confirmation) {
            DeleteConfirmation.VOICE_COMMANDS -> Triple(
                "Voice Commands",
                "~534 MB",
                { viewModel.deleteVoiceCommands() }
            )
            DeleteConfirmation.LOCAL_AI -> Triple(
                "Local AI",
                "~1.3 GB",
                { viewModel.deleteLocalAi() }
            )
        }

        AlertDialog(
            onDismissRequest = { deleteConfirmation = null },
            title = { Text("Delete $title?") },
            text = { Text("This will delete the downloaded model ($size). You can re-download it later.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onConfirm()
                        deleteConfirmation = null
                    }
                ) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { deleteConfirmation = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("SimpleAI") },
                actions = {
                    IconButton(onClick = onNavigateToAbout) {
                        Icon(
                            Icons.Default.Info,
                            contentDescription = "About"
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Voice Commands capability
            CapabilityCard(
                title = "Voice Commands",
                icon = "\uD83C\uDFA4",  // microphone
                description = "Intent classification and entity extraction",
                status = state.voiceCommandsStatus,
                onDownload = { viewModel.downloadVoiceCommands() },
                onRetry = { viewModel.downloadVoiceCommands() },
                onDelete = if (state.voiceCommandsStatus is CapabilityStatus.Ready) {
                    { deleteConfirmation = DeleteConfirmation.VOICE_COMMANDS }
                } else null
            )

            // Translation capability
            CapabilityCard(
                title = "Translation",
                icon = "\uD83C\uDF10",  // globe
                description = "On-device translation between languages",
                status = state.translationStatus,
                onConfigure = onNavigateToTranslationLanguages,
                onTest = if (state.downloadedLanguages.size >= 2) {
                    onNavigateToTranslationTest
                } else null,
                extraContent = {
                    if (state.downloadedLanguages.isNotEmpty()) {
                        val languageNames = state.downloadedLanguages
                            .map { getLanguageName(it) }
                            .sorted()
                            .joinToString(", ")
                        Text(
                            text = "Languages: $languageNames",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            )

            // Cloud AI capability
            CapabilityCard(
                title = "Cloud AI",
                icon = "\u2601\uFE0F",  // cloud
                description = "Cloud-based LLM (requires client auth)",
                status = state.cloudAiStatus,
                extraContent = {
                    if (state.cloudAiStatus is CapabilityStatus.Ready) {
                        Text(
                            text = "No download required",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            )

            // Local AI capability
            CapabilityCard(
                title = "Local AI",
                icon = "\uD83E\uDD16",  // robot
                description = "On-device LLM for offline use",
                status = state.localAiStatus,
                onDownload = { viewModel.downloadLocalAi() },
                onRetry = { viewModel.downloadLocalAi() },
                onDelete = if (state.localAiStatus is CapabilityStatus.Ready) {
                    { deleteConfirmation = DeleteConfirmation.LOCAL_AI }
                } else null
            )

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

private fun getLanguageName(code: String): String {
    return when (code) {
        "af" -> "Afrikaans"
        "ar" -> "Arabic"
        "be" -> "Belarusian"
        "bg" -> "Bulgarian"
        "bn" -> "Bengali"
        "ca" -> "Catalan"
        "cs" -> "Czech"
        "cy" -> "Welsh"
        "da" -> "Danish"
        "de" -> "German"
        "el" -> "Greek"
        "en" -> "English"
        "eo" -> "Esperanto"
        "es" -> "Spanish"
        "et" -> "Estonian"
        "fa" -> "Persian"
        "fi" -> "Finnish"
        "fr" -> "French"
        "ga" -> "Irish"
        "gl" -> "Galician"
        "gu" -> "Gujarati"
        "he" -> "Hebrew"
        "hi" -> "Hindi"
        "hr" -> "Croatian"
        "ht" -> "Haitian Creole"
        "hu" -> "Hungarian"
        "id" -> "Indonesian"
        "is" -> "Icelandic"
        "it" -> "Italian"
        "ja" -> "Japanese"
        "ka" -> "Georgian"
        "kn" -> "Kannada"
        "ko" -> "Korean"
        "lt" -> "Lithuanian"
        "lv" -> "Latvian"
        "mk" -> "Macedonian"
        "mr" -> "Marathi"
        "ms" -> "Malay"
        "mt" -> "Maltese"
        "nl" -> "Dutch"
        "no" -> "Norwegian"
        "pl" -> "Polish"
        "pt" -> "Portuguese"
        "ro" -> "Romanian"
        "ru" -> "Russian"
        "sk" -> "Slovak"
        "sl" -> "Slovenian"
        "sq" -> "Albanian"
        "sv" -> "Swedish"
        "sw" -> "Swahili"
        "ta" -> "Tamil"
        "te" -> "Telugu"
        "th" -> "Thai"
        "tl" -> "Tagalog"
        "tr" -> "Turkish"
        "uk" -> "Ukrainian"
        "ur" -> "Urdu"
        "vi" -> "Vietnamese"
        "zh" -> "Chinese"
        else -> code.uppercase()
    }
}
