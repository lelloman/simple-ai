package com.lelloman.simpleai.ui

import com.lelloman.simpleai.R

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
    val strings = androidx.compose.ui.platform.LocalContext.current
    val state by viewModel.state.collectAsState()
    var showDownloadSettings by remember { mutableStateOf(false) }
    var deleteConfirmation by remember { mutableStateOf<DeleteConfirmation?>(null) }

    // Delete confirmation dialog
    deleteConfirmation?.let { confirmation ->
        val (title, size, onConfirm) = when (confirmation) {
            DeleteConfirmation.VOICE_COMMANDS -> Triple(
                strings.getString(R.string.ui_voice_commands),
                formatSize(com.lelloman.simpleai.model.NluModel.SIZE_BYTES),
                { viewModel.deleteVoiceCommands() }
            )
            DeleteConfirmation.LOCAL_AI -> Triple(
                strings.getString(R.string.ui_local_ai),
                formatSize(com.lelloman.simpleai.model.LocalAIModel.SIZE_BYTES),
                { viewModel.deleteLocalAi() }
            )
        }

        AlertDialog(
            onDismissRequest = { deleteConfirmation = null },
            title = { Text(strings.getString(R.string.ui_delete, title)) },
            text = { Text(strings.getString(R.string.ui_this_will_delete_the_downloaded_model_you_can_re_download_it_late, size)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        onConfirm()
                        deleteConfirmation = null
                    }
                ) {
                    Text(strings.getString(R.string.ui_delete_2), color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { deleteConfirmation = null }) {
                    Text(strings.getString(R.string.ui_cancel))
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(strings.getString(R.string.ui_simpleai)) },
                actions = {
                    IconButton(onClick = onNavigateToAbout) {
                        Icon(
                            Icons.Default.Info,
                            contentDescription = strings.getString(R.string.ui_about)
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
            Text(strings.getString(R.string.ui_ai_for_your_apps), style = MaterialTheme.typography.titleLarge)
            Text(strings.getString(R.string.ui_simpleai_manages_shared_ai_models_for_compatible_apps_download_on))
            Text(strings.getString(R.string.ui_1_enable_simpleai_in_your_compatible_app_n2_download_its_required))
            Text(strings.getString(R.string.ui_you_can_try_translation_here_after_downloading_a_language_pack_vo))
            Text(if (state.isServiceConnected) strings.getString(R.string.ui_service_connected) else strings.getString(R.string.ui_service_disconnected), style = MaterialTheme.typography.bodySmall)
            state.serviceError?.let { error ->
                Text(error, color = MaterialTheme.colorScheme.error)
                TextButton(onClick = viewModel::refreshCapabilities) { Text(strings.getString(R.string.ui_retry_connection)) }
            }
            TextButton(onClick = { showDownloadSettings = !showDownloadSettings }) { Text(strings.getString(R.string.ui_download_and_storage_settings)) }
            if (showDownloadSettings) {
            Text(strings.getString(R.string.ui_download_network_and_sizes, formatSize(com.lelloman.simpleai.model.NluModel.SIZE_BYTES), formatSize(com.lelloman.simpleai.model.LocalAIModel.SIZE_BYTES)))
            Text(strings.getString(R.string.ui_allow_mobile_data_for_new_downloads_charges_may_apply))
            Switch(checked = state.allowMeteredDownloads, onCheckedChange = viewModel::setAllowMeteredDownloads)
            Text(strings.getString(R.string.ui_working_copy_and_reserve, formatSize(com.lelloman.simpleai.model.NluModel.SIZE_BYTES), formatSize(com.lelloman.simpleai.download.DownloadPolicy.RESERVE_BYTES)))
            }
            state.storage?.let { storage ->
                Text(strings.getString(R.string.ui_app_data_available, formatSize(storage.usedBytes), formatSize(storage.availableBytes)))
                Text(strings.getString(R.string.ui_includes_models_language_packs_partial_downloads_and_supporting_a), style = MaterialTheme.typography.bodySmall)
            }
            ConnectedApps()
            // Voice Commands capability
            CapabilityCard(
                title = strings.getString(R.string.ui_voice_commands),
                icon = "\uD83C\uDFA4",  // microphone
                description = strings.getString(R.string.ui_understand_commands_sent_by_your_app_on_this_device),
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
                title = strings.getString(R.string.ui_translation),
                icon = "\uD83C\uDF10",  // globe
                description = strings.getString(R.string.ui_on_device_translation_between_languages),
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
                            text = strings.getString(R.string.ui_languages, languageNames),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            )

            // Cloud AI capability
            CapabilityCard(
                title = strings.getString(R.string.ui_cloud_ai),
                icon = "\u2601\uFE0F",  // cloud
                description = strings.getString(R.string.ui_online_answers_through_your_connected_app_s_account),
                status = state.cloudAiStatus,
                extraContent = {
                    if (state.cloudAiStatus is CapabilityStatus.Ready) {
                        Text(
                            text = strings.getString(R.string.ui_configured_requests_require_internet_and_authorization_from_the_c),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            )

            // Local AI capability
            CapabilityCard(
                title = strings.getString(R.string.ui_local_ai),
                icon = "\uD83E\uDD16",  // robot
                description = strings.getString(R.string.ui_generate_answers_on_this_device_after_downloading_the_model),
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

private fun getLanguageName(code: String): String = com.lelloman.simpleai.translation.Language.fromCode(code)?.displayName ?: code.uppercase(java.util.Locale.ROOT)
