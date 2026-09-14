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
import androidx.compose.material3.Switch
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
    var showDownloadSettings by remember { mutableStateOf(false) }
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
            Text("AI for your apps", style = MaterialTheme.typography.titleLarge)
            Text("SimpleAI manages shared AI models for compatible apps. Download only the features your app needs.")
            Text("1. Enable SimpleAI in your compatible app.\n2. Download its required models here.\n3. Approve the app in Connected apps, then retry there.")
            Text("You can try translation here after downloading a language pack. Voice Commands and AI chat are used from compatible apps; this screen does not record speech or provide a chat conversation.")
            Text(if (state.isServiceConnected) "Service connected" else "Service disconnected", style = MaterialTheme.typography.bodySmall)
            state.serviceError?.let { error ->
                Text(error, color = MaterialTheme.colorScheme.error)
                TextButton(onClick = viewModel::refreshCapabilities) { Text("Retry connection") }
            }
            TextButton(onClick = { showDownloadSettings = !showDownloadSettings }) { Text("Download and storage settings") }
            if (showDownloadSettings) {
            Text("Downloads use unmetered networks by default; language packs require Wi-Fi. Model transfers use about 0.5 GB for Voice Commands and 1.3 GB for Local AI.")
            Text("Allow mobile data for new downloads (charges may apply)")
            Switch(checked = state.allowMeteredDownloads, onCheckedChange = viewModel::setAllowMeteredDownloads)
            Text("Voice Commands also needs a working copy of about 0.5 GB. Downloads reserve 64 MiB of free space.")
            }
            state.storage?.let { storage ->
                Text("App data: ${storage.usedBytes / (1024 * 1024)} MiB • Available: ${storage.availableBytes / (1024 * 1024)} MiB")
                Text("Includes models, language packs, partial downloads and supporting app data.", style = MaterialTheme.typography.bodySmall)
            }
            ConnectedApps()
            // Voice Commands capability
            CapabilityCard(
                title = "Voice Commands",
                icon = "\uD83C\uDFA4",  // microphone
                description = "Understand commands sent by your app, on this device",
                status = state.voiceCommandsStatus,
                downloadJob = state.downloadJobs["voice"],
                onPause = { viewModel.pauseDownload("voice") },
                onDownload = { viewModel.downloadVoiceCommands() },
                onRetry = { viewModel.downloadVoiceCommands() },
                onDelete = if (state.voiceCommandsStatus is CapabilityStatus.Ready || state.voiceCommandsStatus == CapabilityStatus.Downloaded || state.downloadJobs["voice"] in listOf("CANCELLED", "FAILED")) {
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
                onTest = if (state.downloadedLanguages.isNotEmpty()) {
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
                description = "Online answers through your connected app’s account",
                status = state.cloudAiStatus,
                extraContent = {
                    if (state.cloudAiStatus is CapabilityStatus.Ready) {
                        Text(
                            text = "Configured. Requests require internet and authorization from the client app.",
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
                description = "Generate answers on this device after downloading the model",
                status = state.localAiStatus,
                downloadJob = state.downloadJobs["local"],
                onPause = { viewModel.pauseDownload("local") },
                onDownload = { viewModel.downloadLocalAi() },
                onRetry = { viewModel.downloadLocalAi() },
                onDelete = if (state.localAiStatus is CapabilityStatus.Ready || state.localAiStatus == CapabilityStatus.Downloaded || state.downloadJobs["local"] in listOf("CANCELLED", "FAILED")) {
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
