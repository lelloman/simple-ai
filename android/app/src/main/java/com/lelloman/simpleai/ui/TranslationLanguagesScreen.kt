package com.lelloman.simpleai.ui

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
import androidx.compose.ui.Alignment
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
    val flag: String,
    val isDownloaded: Boolean,
    val isDownloading: Boolean = false,
    val isRequired: Boolean = false  // English is required
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
    val state by viewModel.state.collectAsState()
    val downloadedLanguages = state.downloadedLanguages
    val downloadingLanguage = state.downloadingLanguage
    val languageDownloadError = state.languageDownloadError
    val allLanguages = getLanguageInfoList(downloadedLanguages, downloadingLanguage)
    val downloaded = allLanguages.filter { it.isDownloaded || it.isRequired }
    val available = allLanguages.filter { !it.isDownloaded && !it.isRequired }

    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(languageDownloadError) {
        languageDownloadError?.let { error ->
            snackbarHostState.showSnackbar(error)
            viewModel.clearLanguageDownloadError()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Translation Languages") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }
            )
        },
        snackbarHost = {
            SnackbarHost(snackbarHostState) { data ->
                Snackbar(
                    snackbarData = data,
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer
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
            // Downloaded section
            if (downloaded.isNotEmpty()) {
                item {
                    Text(
                        text = "Downloaded",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }

                items(downloaded) { language ->
                    LanguageCard(
                        language = language,
                        onAction = {
                            if (!language.isRequired) {
                                viewModel.deleteTranslationLanguage(language.code)
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
                        text = "Available",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }

                items(available) { language ->
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
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (language.isDownloaded || language.isRequired) {
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
            Text(
                text = language.flag,
                style = MaterialTheme.typography.headlineSmall
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = language.name,
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium
                )
                if (language.isRequired) {
                    Text(
                        text = "Required (pivot language)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else if (!language.isDownloaded) {
                    Text(
                        text = "~30 MB",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            when {
                language.isDownloading -> {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp
                    )
                }
                language.isRequired -> {
                    Icon(
                        Icons.Default.Check,
                        contentDescription = "Required",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                }
                language.isDownloaded -> {
                    IconButton(onClick = onAction) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = "Delete",
                            tint = MaterialTheme.colorScheme.error
                        )
                    }
                }
                else -> {
                    IconButton(onClick = onAction) {
                        Icon(
                            Icons.Default.Add,
                            contentDescription = "Download",
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
    downloadingLanguage: String?
): List<LanguageInfo> {
    return LANGUAGE_DATA.map { (code, name, flag) ->
        LanguageInfo(
            code = code,
            name = name,
            flag = flag,
            isDownloaded = code in downloadedLanguages,
            isDownloading = code == downloadingLanguage,
            isRequired = code == "en" && downloadedLanguages.isNotEmpty()
        )
    }.sortedWith(compareBy({ !it.isRequired }, { !it.isDownloaded }, { it.name }))
}

// Language code -> (name, flag)
private val LANGUAGE_DATA = listOf(
    Triple("af", "Afrikaans", "\uD83C\uDDFF\uD83C\uDDE6"),
    Triple("ar", "Arabic", "\uD83C\uDDF8\uD83C\uDDE6"),
    Triple("be", "Belarusian", "\uD83C\uDDE7\uD83C\uDDFE"),
    Triple("bg", "Bulgarian", "\uD83C\uDDE7\uD83C\uDDEC"),
    Triple("bn", "Bengali", "\uD83C\uDDE7\uD83C\uDDE9"),
    Triple("ca", "Catalan", "\uD83C\uDDEA\uD83C\uDDF8"),
    Triple("cs", "Czech", "\uD83C\uDDE8\uD83C\uDDFF"),
    Triple("cy", "Welsh", "\uD83C\uDFF4\uDB40\uDC67\uDB40\uDC62\uDB40\uDC77\uDB40\uDC6C\uDB40\uDC73\uDB40\uDC7F"),
    Triple("da", "Danish", "\uD83C\uDDE9\uD83C\uDDF0"),
    Triple("de", "German", "\uD83C\uDDE9\uD83C\uDDEA"),
    Triple("el", "Greek", "\uD83C\uDDEC\uD83C\uDDF7"),
    Triple("en", "English", "\uD83C\uDDEC\uD83C\uDDE7"),
    Triple("eo", "Esperanto", "\uD83C\uDDEA\uD83C\uDDFA"),
    Triple("es", "Spanish", "\uD83C\uDDEA\uD83C\uDDF8"),
    Triple("et", "Estonian", "\uD83C\uDDEA\uD83C\uDDEA"),
    Triple("fa", "Persian", "\uD83C\uDDEE\uD83C\uDDF7"),
    Triple("fi", "Finnish", "\uD83C\uDDEB\uD83C\uDDEE"),
    Triple("fr", "French", "\uD83C\uDDEB\uD83C\uDDF7"),
    Triple("ga", "Irish", "\uD83C\uDDEE\uD83C\uDDEA"),
    Triple("gl", "Galician", "\uD83C\uDDEA\uD83C\uDDF8"),
    Triple("gu", "Gujarati", "\uD83C\uDDEE\uD83C\uDDF3"),
    Triple("he", "Hebrew", "\uD83C\uDDEE\uD83C\uDDF1"),
    Triple("hi", "Hindi", "\uD83C\uDDEE\uD83C\uDDF3"),
    Triple("hr", "Croatian", "\uD83C\uDDED\uD83C\uDDF7"),
    Triple("ht", "Haitian Creole", "\uD83C\uDDED\uD83C\uDDF9"),
    Triple("hu", "Hungarian", "\uD83C\uDDED\uD83C\uDDFA"),
    Triple("id", "Indonesian", "\uD83C\uDDEE\uD83C\uDDE9"),
    Triple("is", "Icelandic", "\uD83C\uDDEE\uD83C\uDDF8"),
    Triple("it", "Italian", "\uD83C\uDDEE\uD83C\uDDF9"),
    Triple("ja", "Japanese", "\uD83C\uDDEF\uD83C\uDDF5"),
    Triple("ka", "Georgian", "\uD83C\uDDEC\uD83C\uDDEA"),
    Triple("kn", "Kannada", "\uD83C\uDDEE\uD83C\uDDF3"),
    Triple("ko", "Korean", "\uD83C\uDDF0\uD83C\uDDF7"),
    Triple("lt", "Lithuanian", "\uD83C\uDDF1\uD83C\uDDF9"),
    Triple("lv", "Latvian", "\uD83C\uDDF1\uD83C\uDDFB"),
    Triple("mk", "Macedonian", "\uD83C\uDDF2\uD83C\uDDF0"),
    Triple("mr", "Marathi", "\uD83C\uDDEE\uD83C\uDDF3"),
    Triple("ms", "Malay", "\uD83C\uDDF2\uD83C\uDDFE"),
    Triple("mt", "Maltese", "\uD83C\uDDF2\uD83C\uDDF9"),
    Triple("nl", "Dutch", "\uD83C\uDDF3\uD83C\uDDF1"),
    Triple("no", "Norwegian", "\uD83C\uDDF3\uD83C\uDDF4"),
    Triple("pl", "Polish", "\uD83C\uDDF5\uD83C\uDDF1"),
    Triple("pt", "Portuguese", "\uD83C\uDDF5\uD83C\uDDF9"),
    Triple("ro", "Romanian", "\uD83C\uDDF7\uD83C\uDDF4"),
    Triple("ru", "Russian", "\uD83C\uDDF7\uD83C\uDDFA"),
    Triple("sk", "Slovak", "\uD83C\uDDF8\uD83C\uDDF0"),
    Triple("sl", "Slovenian", "\uD83C\uDDF8\uD83C\uDDEE"),
    Triple("sq", "Albanian", "\uD83C\uDDE6\uD83C\uDDF1"),
    Triple("sv", "Swedish", "\uD83C\uDDF8\uD83C\uDDEA"),
    Triple("sw", "Swahili", "\uD83C\uDDF0\uD83C\uDDEA"),
    Triple("ta", "Tamil", "\uD83C\uDDEE\uD83C\uDDF3"),
    Triple("te", "Telugu", "\uD83C\uDDEE\uD83C\uDDF3"),
    Triple("th", "Thai", "\uD83C\uDDF9\uD83C\uDDED"),
    Triple("tl", "Tagalog", "\uD83C\uDDF5\uD83C\uDDED"),
    Triple("tr", "Turkish", "\uD83C\uDDF9\uD83C\uDDF7"),
    Triple("uk", "Ukrainian", "\uD83C\uDDFA\uD83C\uDDE6"),
    Triple("ur", "Urdu", "\uD83C\uDDF5\uD83C\uDDF0"),
    Triple("vi", "Vietnamese", "\uD83C\uDDFB\uD83C\uDDF3"),
    Triple("zh", "Chinese", "\uD83C\uDDE8\uD83C\uDDF3")
)
