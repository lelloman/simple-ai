package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.lelloman.lellodesign.LelloIconButton
import com.lelloman.lellodesign.LocalLelloPalette
import com.lelloman.simpleai.R

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
        CalmDialog(
            title = strings.getString(R.string.ui_delete, language.name),
            onDismiss = { deleteLanguage = null },
            confirmLabel = strings.getString(R.string.ui_delete_language_pack),
            onConfirm = { viewModel.deleteTranslationLanguage(language.code); deleteLanguage = null },
            dismissLabel = strings.getString(R.string.ui_cancel),
            danger = true,
        ) { Text(strings.getString(R.string.ui_apps_will_need_this_language_pack_downloaded_again_before_transla)) }
    }
    LaunchedEffect(state.languageOperationMessage) {
        state.languageOperationMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearLanguageOperationMessage()
        }
    }

    val builtIn = allLanguages.first { it.isBuiltIn }
    val showBuiltIn = matching.any { it.isBuiltIn }
    Scaffold(
        containerColor = MaterialTheme.colorScheme.surface,
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
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
                .widthIn(max = 720.dp)
                .padding(innerPadding),
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 24.dp),
        ) {
            item {
                OutlinedTextField(query, { query = it }, label = { Text(strings.getString(R.string.ui_search_languages)) },
                    singleLine = true, modifier = Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.small,
                    leadingIcon = { Icon(painterResource(R.drawable.ic_search), null) },
                    trailingIcon = {
                        if (query.isNotEmpty()) LelloIconButton(onClick = { query = "" },
                            contentDescription = strings.getString(R.string.ui_clear)) { Icon(painterResource(R.drawable.ic_close), null) }
                    })
                if (matching.isEmpty()) Text(strings.getString(R.string.ui_no_languages_match_your_search),
                    Modifier.padding(top = 24.dp), style = MaterialTheme.typography.bodyLarge,
                    color = LocalLelloPalette.current["text-secondary"])
            }
            // On this device: English is built in, then downloaded packs
            if (showBuiltIn || downloaded.isNotEmpty()) {
                item {
                    ListSectionHeader(strings.getString(R.string.ui_downloaded),
                        meta = strings.getString(R.string.languages_count, downloaded.size + if (showBuiltIn) 1 else 0))
                }
                if (showBuiltIn) item(key = builtIn.code) {
                    LanguageRow(language = builtIn, onAction = {})
                }
                items(downloaded, key = { it.code }) { language ->
                    LanguageRow(
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
                    Spacer(modifier = Modifier.height(16.dp))
                    ListSectionHeader(strings.getString(R.string.ui_available),
                        meta = strings.getString(R.string.ui_language_pack_estimate,
                            formatSize(com.lelloman.simpleai.capability.CapabilityManager.TRANSLATION_LANGUAGE_SIZE)))
                }

                items(available, key = { it.code }) { language ->
                    LanguageRow(
                        language = language,
                        onAction = { viewModel.downloadTranslationLanguage(language.code) }
                    )
                }
            }
        }
    }
}

@Composable
private fun LanguageRow(
    language: LanguageInfo,
    onAction: () -> Unit
) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    val palette = LocalLelloPalette.current
    Column {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 64.dp)
                .padding(vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = language.name,
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium
                )
                val nativeName = nativeLanguageName(language.code)
                val details = listOfNotNull(
                    nativeName.takeUnless { it.equals(language.name, ignoreCase = true) },
                    if (language.isBuiltIn) strings.getString(R.string.ui_built_in_no_download_needed) else null,
                )
                if (details.isNotEmpty()) Text(details.joinToString(" · "), style = MaterialTheme.typography.bodyMedium,
                    color = palette["text-secondary"])
            }

            when {
                language.isDownloading -> {
                    Box(Modifier.size(48.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp).semantics { contentDescription = strings.getString(R.string.ui_downloading_2, language.name) },
                            strokeWidth = 2.dp
                        )
                    }
                }
                language.isBuiltIn -> Unit
                language.isDownloaded -> {
                    LelloIconButton(onClick = onAction,
                        contentDescription = strings.getString(R.string.ui_delete_language_pack_2, language.name)) {
                        Icon(painterResource(R.drawable.ic_delete), null, tint = palette["text-secondary"])
                    }
                }
                else -> {
                    LelloIconButton(onClick = onAction,
                        contentDescription = strings.getString(R.string.ui_download_language_pack, language.name)) {
                        Icon(painterResource(R.drawable.ic_download), null, tint = MaterialTheme.colorScheme.primary)
                    }
                }
            }
        }
        RowDivider()
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
