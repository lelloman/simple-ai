package com.lelloman.simpleai.ui

import com.lelloman.simpleai.R

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.semantics.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Language info for display.
 */
data class LanguageInfo(
    val code: String,
    val name: String,
    val isDownloaded: Boolean,
    val isDownloading: Boolean = false,
    val isBuiltIn: Boolean = false  // English is required
)

/**
 * Screen for managing translation languages.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranslationLanguagesScreen(
    viewModel: CapabilitiesViewModel = viewModel(),
    onBack: () -> Unit
) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    val state by viewModel.state.collectAsState()
    val downloadedLanguages = state.downloadedLanguages
    val downloadingLanguages = state.downloadingLanguages
    val languageDownloadError = state.languageDownloadErrors.entries.firstOrNull()
    var query by rememberSaveable { mutableStateOf("") }
    var deleteLanguage by remember { mutableStateOf<LanguageInfo?>(null) }
    val allLanguages = getLanguageInfoList(downloadedLanguages, downloadingLanguages)
    val matching = allLanguages.filter { languageMatches(it.code, it.name, query) }
    val downloaded = matching.filter { it.isDownloaded && !it.isBuiltIn }
    val available = matching.filter { !it.isDownloaded && !it.isBuiltIn }

    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(state.languageOperationError) {
        state.languageOperationError?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearLanguageOperationError()
        }
    }
    LaunchedEffect(languageDownloadError) {
        languageDownloadError?.let { error ->
            snackbarHostState.showSnackbar(strings.getString(R.string.ui_value_4, error.key, error.value))
            viewModel.clearLanguageDownloadError(error.key)
        }
    }

    deleteLanguage?.let { language ->
        AlertDialog(
            onDismissRequest = { deleteLanguage = null },
            title = { Text(strings.getString(R.string.ui_delete, language.name)) },
            text = { Text(strings.getString(R.string.ui_apps_will_need_this_language_pack_downloaded_again_before_transla)) },
            confirmButton = { TextButton(onClick = { viewModel.deleteTranslationLanguage(language.code); deleteLanguage = null }) { Text(strings.getString(R.string.ui_delete_language_pack)) } },
            dismissButton = { TextButton(onClick = { deleteLanguage = null }) { Text(strings.getString(R.string.ui_cancel)) } }
        )
    }
    LaunchedEffect(state.languageOperationMessage) {
        state.languageOperationMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearLanguageOperationMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(strings.getString(R.string.ui_manage_languages)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = strings.getString(R.string.ui_back)
                        )
                    }
                }
            )
        },
        snackbarHost = {
            SnackbarHost(snackbarHostState) { data ->
                Snackbar(
                    snackbarData = data,
                )
            }
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            item {
                OutlinedTextField(query, { query = it }, label = { Text(strings.getString(R.string.ui_search_languages)) }, singleLine = true, modifier = Modifier.fillMaxWidth(), trailingIcon = {
                    if (query.isNotEmpty()) TextButton(onClick = { query = "" }) { Text(strings.getString(R.string.ui_clear)) }
                })
                if (matching.isEmpty()) Text(strings.getString(R.string.ui_no_languages_match_your_search))
            }
            if (matching.any { it.isBuiltIn }) item {
                LanguageCard(language = allLanguages.first { it.isBuiltIn }, onAction = {})
            }
            // Downloaded section
            if (downloaded.isNotEmpty()) {
                item {
                    Text(
                        text = strings.getString(R.string.ui_downloaded),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }

                items(downloaded, key = { it.code }) { language ->
                    LanguageCard(
                        language = language,
                        onAction = {
                            if (!language.isBuiltIn) {
                                deleteLanguage = language
                            }
                        }
                    )
                }
            }

            // Available section
            if (available.isNotEmpty()) {
                item {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = strings.getString(R.string.ui_available),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }

                items(available, key = { it.code }) { language ->
                    LanguageCard(
                        language = language,
                        onAction = { viewModel.downloadTranslationLanguage(language.code) }
                    )
                }
            }

            // Bottom padding
            item {
                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

@Composable
private fun LanguageCard(
    language: LanguageInfo,
    onAction: () -> Unit
) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (language.isDownloaded || language.isBuiltIn) {
                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = language.name,
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium
                )
                val nativeName = nativeLanguageName(language.code)
                if (!nativeName.equals(language.name, ignoreCase = true)) Text(nativeName, style = MaterialTheme.typography.bodySmall)
                if (language.isBuiltIn) {
                    Text(
                        text = strings.getString(R.string.ui_built_in_no_download_needed),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else if (!language.isDownloaded) {
                    Text(
                        text = strings.getString(R.string.ui_language_pack_estimate, formatSize(com.lelloman.simpleai.capability.CapabilityManager.TRANSLATION_LANGUAGE_SIZE)),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            when {
                language.isDownloading -> {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp).semantics { contentDescription = strings.getString(R.string.ui_downloading_2, language.name) },
                        strokeWidth = 2.dp
                    )
                }
                language.isBuiltIn -> {
                    Icon(
                        Icons.Default.Check,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                }
                language.isDownloaded -> {
                    IconButton(onClick = onAction) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = strings.getString(R.string.ui_delete_language_pack_2, language.name),
                            tint = MaterialTheme.colorScheme.error
                        )
                    }
                }
                else -> {
                    IconButton(onClick = onAction) {
                        Icon(
                            Icons.Default.Add,
                            contentDescription = strings.getString(R.string.ui_download_language_pack, language.name),
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        }
    }
}

/**
 * Get list of all supported languages with their info.
 */
private fun getLanguageInfoList(
    downloadedLanguages: Set<String>,
    downloadingLanguages: Set<String>
): List<LanguageInfo> {
    return com.lelloman.simpleai.translation.Language.entries.map { language ->
        val code = language.code
        LanguageInfo(
            code = code,
            name = language.displayName,
            isDownloaded = code in downloadedLanguages,
            isDownloading = code in downloadingLanguages,
            isBuiltIn = code == "en"
        )
    }.sortedWith(compareBy({ !it.isBuiltIn }, { !it.isDownloaded }, { it.name }))
}
