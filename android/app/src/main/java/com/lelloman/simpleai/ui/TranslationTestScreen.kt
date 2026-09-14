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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.material3.TextButton
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
    val strings = androidx.compose.ui.platform.LocalContext.current
    val clipboard = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }
    val state by viewModel.state.collectAsState()
    val translationState by viewModel.translationState.collectAsState()
    val downloadedLanguages = com.lelloman.simpleai.translation.TranslationAvailability.available(state.downloadedLanguages)
    androidx.compose.runtime.LaunchedEffect(translationState.translatedText) { copied = false }
    val draft = translationState.draft
    val inputText = draft.text
    val sourceLang = draft.source
    val targetLang = draft.target

    val languageOptions = listOf("auto" to strings.getString(R.string.ui_auto_detect)) +
        downloadedLanguages.sorted().map { it to getLanguageDisplayName(it) }

    val targetOptions = downloadedLanguages.sorted().map { it to getLanguageDisplayName(it) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(strings.getString(R.string.ui_test_translation)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = strings.getString(R.string.ui_back)
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
            Text(strings.getString(R.string.ui_this_test_sends_a_request_through_the_simpleai_service_to_check_a), style = MaterialTheme.typography.bodySmall)
            // Language selection row
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Source language dropdown
                LanguageDropdown(
                    label = strings.getString(R.string.ui_from),
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
                    modifier = Modifier.semantics { contentDescription = strings.getString(R.string.ui_swap_source_and_target_languages) },
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
                    label = strings.getString(R.string.ui_to),
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
                label = { Text(strings.getString(R.string.ui_enter_text_to_translate)) },
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
                Text(if (translationState.isTranslating) strings.getString(R.string.ui_translating) else strings.getString(R.string.ui_translate))
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
                                text = strings.getString(R.string.ui_error),
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
                                    text = strings.getString(R.string.ui_detected, getLanguageDisplayName(detectedLang)),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                            }
                            Text(
                                text = strings.getString(R.string.ui_translation),
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            SelectionContainer {
                                Text(translationState.translatedText ?: "", style = MaterialTheme.typography.bodyLarge)
                            }
                            TextButton(onClick = {
                                clipboard.setText(AnnotatedString(translationState.translatedText.orEmpty()))
                                copied = true
                            }) { Text(strings.getString(R.string.ui_copy_translation)) }
                            if (copied) Text(strings.getString(R.string.ui_copied), modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
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
    val strings = androidx.compose.ui.platform.LocalContext.current
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

private fun getLanguageDisplayName(code: String): String = com.lelloman.simpleai.translation.Language.fromCode(code)?.displayName ?: code.uppercase(java.util.Locale.ROOT)
