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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.semantics.*
import androidx.compose.foundation.layout.heightIn
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Screen for testing translation functionality.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranslationTestScreen(
    viewModel: CapabilitiesViewModel = viewModel(),
    onBack: () -> Unit
) {
    val state by viewModel.state.collectAsState()
    val translationState by viewModel.translationState.collectAsState()
    val downloadedLanguages = com.lelloman.simpleai.translation.TranslationAvailability.available(state.downloadedLanguages)
    val draft = translationState.draft
    val inputText = draft.text
    val sourceLang = draft.source
    val targetLang = draft.target

    val languageOptions = listOf("auto" to "Auto-detect") +
        downloadedLanguages.sorted().map { it to getLanguageDisplayName(it) }

    val targetOptions = downloadedLanguages.sorted().map { it to getLanguageDisplayName(it) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Test Translation") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
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
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text("This test sends a request through the SimpleAI service. To check another app’s setup, also try a request from that app.", style = MaterialTheme.typography.bodySmall)
            // Language selection row
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Source language dropdown
                LanguageDropdown(
                    label = "From",
                    selectedCode = sourceLang,
                    options = languageOptions,
                    onSelect = { viewModel.editTranslation(draft.copy(source = it)) },
                    modifier = Modifier.fillMaxWidth()
                )

                // Swap button
                IconButton(
                    onClick = {
                        if (sourceLang != "auto") {
                            viewModel.editTranslation(draft.copy(source = targetLang, target = sourceLang))
                        }
                    },
                    modifier = Modifier.semantics { contentDescription = "Swap source and target languages" },
                    enabled = sourceLang != "auto"
                ) {
                    Text(
                        modifier = Modifier.clearAndSetSemantics {},
                        text = "\u21C4",  // Unicode arrows for swap
                        style = MaterialTheme.typography.titleLarge
                    )
                }

                // Target language dropdown
                LanguageDropdown(
                    label = "To",
                    selectedCode = targetLang,
                    options = targetOptions,
                    onSelect = { viewModel.editTranslation(draft.copy(target = it)) },
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Input text field
            OutlinedTextField(
                value = inputText,
                onValueChange = { viewModel.editTranslation(draft.copy(text = it)) },
                label = { Text("Enter text to translate") },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 150.dp),
                minLines = 4,
                maxLines = 12
            )

            // Translate button
            Button(
                onClick = { viewModel.translate() },
                enabled = inputText.isNotBlank() && !translationState.isTranslating,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (translationState.isTranslating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(if (translationState.isTranslating) "Translating..." else "Translate")
            }

            // Result card
            if (translationState.translatedText != null || translationState.error != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (translationState.error != null) {
                            MaterialTheme.colorScheme.errorContainer
                        } else {
                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                        }
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                    ) {
                        val errorMessage = translationState.error
                        val detectedLang = translationState.detectedLanguage
                        if (errorMessage != null) {
                            Text(
                                text = "Error",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.error,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = errorMessage,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        } else {
                            if (detectedLang != null && sourceLang == "auto") {
                                Text(
                                    text = "Detected: ${getLanguageDisplayName(detectedLang)}",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                            }
                            Text(
                                text = "Translation",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = translationState.translatedText ?: "",
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguageDropdown(
    label: String,
    selectedCode: String,
    options: List<Pair<String, String>>,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedName = options.find { it.first == selectedCode }?.second ?: selectedCode

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = modifier
    ) {
        OutlinedTextField(
            value = selectedName,
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                .fillMaxWidth(),
            singleLine = true
        )

        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            options.forEach { (code, name) ->
                DropdownMenuItem(
                    text = { Text(name) },
                    onClick = {
                        onSelect(code)
                        expanded = false
                    }
                )
            }
        }
    }
}

private fun getLanguageDisplayName(code: String): String {
    return when (code) {
        "auto" -> "Auto-detect"
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
