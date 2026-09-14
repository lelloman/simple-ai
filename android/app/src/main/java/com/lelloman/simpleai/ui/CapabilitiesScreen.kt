package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
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
    Scaffold(topBar = { TopAppBar(title = { Text(stringResource(R.string.nav_models)) }) }) { padding ->
        ModelsContent(state, onOpenModel, onNavigateToTranslationLanguages, Modifier.padding(padding))
    }
}

@Composable
internal fun ModelsContent(state: CapabilitiesState, onOpenModel: (String) -> Unit, onLanguages: () -> Unit, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        ModelRow(stringResource(R.string.ui_voice_commands), modelSummary(state.voiceCommandsStatus, state.downloadJobs["voice"], NluModel.SIZE_BYTES), { onOpenModel("voice") })
        ModelRow(stringResource(R.string.ui_local_ai), modelSummary(state.localAiStatus, state.downloadJobs["local"], LocalAIModel.SIZE_BYTES), { onOpenModel("local") })
        ModelRow(stringResource(R.string.model_languages), if (state.downloadedLanguages.isEmpty()) stringResource(R.string.model_add_languages) else stringResource(R.string.model_language_count, state.downloadedLanguages.size), onLanguages)
    }
}

@Composable
private fun modelSummary(status: CapabilityStatus, job: String?, size: Long): String = when {
    status == CapabilityStatus.Ready || status == CapabilityStatus.Downloaded -> stringResource(R.string.model_downloaded)
    job == "ENQUEUED" || job == "BLOCKED" -> stringResource(R.string.model_queued)
    status is CapabilityStatus.Downloading -> stringResource(R.string.ui_downloading, (status.progress * 100).toInt())
    job == "CANCELLED" && status is CapabilityStatus.Error -> stringResource(R.string.model_paused)
    status is CapabilityStatus.Error -> stringResource(R.string.model_needs_attention)
    status == CapabilityStatus.Checking || status == CapabilityStatus.Loading -> stringResource(R.string.model_checking)
    else -> formatSize(size)
}

@Composable
internal fun ModelRow(title: String, summary: String, onClick: () -> Unit) {
    OutlinedCard(onClick = onClick, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(summary, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, contentDescription = null)
        }
    }
}
