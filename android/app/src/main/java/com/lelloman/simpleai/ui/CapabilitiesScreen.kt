package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import com.lelloman.lellodesign.LelloSection
import com.lelloman.lellodesign.LelloWorkspace
import com.lelloman.simpleai.R
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.model.NluModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CapabilitiesScreen(
    viewModel: CapabilitiesViewModel,
    onNavigateToTranslationLanguages: () -> Unit,
    onOpenModel: (String) -> Unit
) {
    val state by viewModel.state.collectAsState()
    ModelsContent(state, onOpenModel, onNavigateToTranslationLanguages)
}

@Composable
internal fun ModelsContent(state: CapabilitiesState, onOpenModel: (String) -> Unit, onLanguages: () -> Unit, modifier: Modifier = Modifier) {
    LelloWorkspace(modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        LelloSection(stringResource(R.string.models_on_device), meta = stringResource(R.string.models_on_device_meta)) {
            Column {
                ModelRow(stringResource(R.string.ui_voice_commands), state.voiceCommandsStatus, state.downloadJobs["voice"],
                    NluModel.SIZE_BYTES) { onOpenModel("voice") }
                RowDivider()
                ModelRow(stringResource(R.string.ui_local_ai), state.localAiStatus, state.downloadJobs["local"],
                    LocalAIModel.SIZE_BYTES) { onOpenModel("local") }
            }
        }
        LelloSection(stringResource(R.string.models_translation)) {
            NavigationRow(
                stringResource(R.string.model_languages),
                if (state.downloadedLanguages.isEmpty()) stringResource(R.string.model_add_languages)
                else stringResource(R.string.model_language_count, state.downloadedLanguages.size),
                onLanguages,
            )
        }
    }
}

/** Status shown beside a model: null while the model is simply not downloaded. */
internal data class ModelStatus(val label: String, val tone: PillTone)

@Composable
internal fun modelStatus(status: CapabilityStatus, job: String?): ModelStatus? = when {
    status == CapabilityStatus.Ready || status == CapabilityStatus.Downloaded -> ModelStatus(stringResource(R.string.model_downloaded), PillTone.Success)
    job == "ENQUEUED" || job == "BLOCKED" -> ModelStatus(stringResource(R.string.model_queued), PillTone.Neutral)
    status is CapabilityStatus.Downloading -> ModelStatus(stringResource(R.string.ui_downloading, (status.progress * 100).toInt()), PillTone.Info)
    job == "CANCELLED" && status is CapabilityStatus.Error -> ModelStatus(stringResource(R.string.model_paused), PillTone.Warning)
    status is CapabilityStatus.Error -> ModelStatus(stringResource(R.string.model_needs_attention), PillTone.Error)
    status == CapabilityStatus.Checking || status == CapabilityStatus.Loading -> ModelStatus(stringResource(R.string.model_checking), PillTone.Neutral)
    else -> null
}

@Composable
private fun ModelRow(title: String, status: CapabilityStatus, job: String?, size: Long, onClick: () -> Unit) {
    val summary = modelStatus(status, job)
    NavigationRow(
        title,
        stringResource(R.string.model_download_size, formatSize(size)),
        onClick,
        status = summary?.let { { StatusPill(it.label, it.tone) } },
    )
}
