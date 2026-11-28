package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.lelloman.lellodesign.LelloAlert
import com.lelloman.lellodesign.LelloButton
import com.lelloman.lellodesign.LelloDimensions
import com.lelloman.lellodesign.LelloIconButton
import com.lelloman.lellodesign.LelloIllustration
import com.lelloman.lellodesign.LelloIllustrationKind
import com.lelloman.lellodesign.LelloLoader
import com.lelloman.lellodesign.LelloSection
import com.lelloman.lellodesign.LelloState
import com.lelloman.lellodesign.LelloTextButton
import com.lelloman.lellodesign.LelloTextStyles
import com.lelloman.lellodesign.LelloTone
import com.lelloman.lellodesign.LelloWorkspace
import com.lelloman.simpleai.R
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.translation.TranslationAvailability

/**
 * Screen for testing translation functionality: a utility flow of languages, input, one primary
 * action and the result with its copy action.
 */
@Composable
fun TranslationTestScreen(
    viewModel: CapabilitiesViewModel = viewModel(),
    @Suppress("UNUSED_PARAMETER") onBack: (() -> Unit)? = null,
    onLanguages: () -> Unit = {}
) {
    val state by viewModel.state.collectAsState()
    val translationState by viewModel.translationState.collectAsState()
    TranslationContent(
        downloadedLanguages = state.downloadedLanguages,
        checking = state.translationStatus == CapabilityStatus.Checking,
        translation = translationState,
        onEdit = viewModel::editTranslation,
        onTranslate = viewModel::translate,
        onLanguages = onLanguages,
    )
}

@Composable
internal fun TranslationContent(
    downloadedLanguages: Set<String>,
    checking: Boolean,
    translation: TranslationState,
    onEdit: (TranslationDraft) -> Unit,
    onTranslate: () -> Unit,
    onLanguages: () -> Unit,
) {
    if (downloadedLanguages.isEmpty()) {
        LelloWorkspace(Modifier.fillMaxSize()) {
            if (checking) {
                LelloLoader(stringResource(R.string.model_checking))
            } else {
                LelloState(
                    title = stringResource(R.string.translate_empty_title),
                    description = stringResource(R.string.translate_empty_hint),
                    modifier = Modifier.fillMaxWidth(),
                    illustration = { LelloIllustration(LelloIllustrationKind.Empty) },
                    action = { LelloButton(onClick = onLanguages) { Text(stringResource(R.string.ui_download_languages)) } },
                )
            }
        }
        return
    }
    val clipboard = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }
    LaunchedEffect(translation.translatedText) { copied = false }
    val draft = translation.draft
    val available = TranslationAvailability.available(downloadedLanguages).sorted()
    val autoDetect = stringResource(R.string.ui_auto_detect)
    val sourceOptions = listOf("auto" to autoDetect) + available.map { it to getLanguageDisplayName(it) }
    val targetOptions = available.map { it to getLanguageDisplayName(it) }

    LelloWorkspace(
        modifier = Modifier
            .widthIn(max = 640.dp)
            .fillMaxSize()
            .imePadding()
            .verticalScroll(rememberScrollState())
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(LelloDimensions.space2)) {
            LanguagePair(
                source = { LanguageDropdown(stringResource(R.string.ui_from), draft.source, sourceOptions,
                    { onEdit(draft.copy(source = it)) }, it) },
                target = { LanguageDropdown(stringResource(R.string.ui_to), draft.target, targetOptions,
                    { onEdit(draft.copy(target = it)) }, it) },
                swap = {
                    LelloIconButton(
                        onClick = { if (draft.source != "auto") onEdit(draft.copy(source = draft.target, target = draft.source)) },
                        contentDescription = stringResource(R.string.ui_swap_source_and_target_languages),
                        enabled = draft.source != "auto",
                    ) { Icon(painterResource(R.drawable.ic_swap), null) }
                },
            )
            LelloTextButton(onClick = onLanguages) { Text(stringResource(R.string.ui_manage_languages)) }
        }

        OutlinedTextField(
            value = draft.text,
            onValueChange = { onEdit(draft.copy(text = it)) },
            label = { Text(stringResource(R.string.ui_enter_text_to_translate)) },
            modifier = Modifier.fillMaxWidth().heightIn(min = 140.dp),
            shape = MaterialTheme.shapes.small,
            minLines = 4,
            maxLines = 12
        )

        LelloButton(
            onClick = onTranslate,
            enabled = draft.text.isNotBlank() && !translation.isTranslating,
            modifier = Modifier.fillMaxWidth()
        ) {
            if (translation.isTranslating) {
                CircularProgressIndicator(
                    modifier = Modifier.size(18.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onPrimary
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(stringResource(if (translation.isTranslating) R.string.ui_translating else R.string.ui_translate))
        }

        val errorMessage = translation.error
        val result = translation.translatedText
        if (errorMessage != null) {
            LelloAlert(stringResource(R.string.translate_failed), tone = LelloTone.Error) { Text(errorMessage) }
        } else if (result != null) {
            val detected = translation.detectedLanguage
            LelloSection(
                stringResource(R.string.ui_translation),
                meta = if (detected != null && draft.source == "auto")
                    stringResource(R.string.ui_detected, getLanguageDisplayName(detected)) else null,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(LelloDimensions.space2)) {
                    SelectionContainer {
                        Text(result, style = MaterialTheme.typography.bodyLarge)
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        LelloTextButton(onClick = {
                            clipboard.setText(AnnotatedString(result))
                            copied = true
                        }) {
                            Icon(painterResource(R.drawable.ic_copy), null, Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(stringResource(R.string.ui_copy_translation))
                        }
                        if (copied) Text(stringResource(R.string.ui_copied), style = LelloTextStyles.caption,
                            modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
                    }
                }
            }
        }
    }
}

/** Source and target side by side with the swap control between them; stacked when narrow. */
@Composable
private fun LanguagePair(
    source: @Composable (Modifier) -> Unit,
    target: @Composable (Modifier) -> Unit,
    swap: @Composable () -> Unit,
) {
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        if (maxWidth >= 360.dp) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                source(Modifier.weight(1f))
                swap()
                target(Modifier.weight(1f))
            }
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                source(Modifier.fillMaxWidth())
                swap()
                target(Modifier.fillMaxWidth())
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
            shape = MaterialTheme.shapes.small,
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
