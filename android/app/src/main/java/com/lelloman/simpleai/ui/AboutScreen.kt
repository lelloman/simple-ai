package com.lelloman.simpleai.ui

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.R

@Composable
fun AboutScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val uri = LocalUriHandler.current
    var help by remember { mutableStateOf(false) }
    var licenses by remember { mutableStateOf(false) }
    SimplePage(stringResource(R.string.ui_about), onBack) {
        Text(stringResource(R.string.ui_simpleai), style = MaterialTheme.typography.headlineLarge)
        Text(stringResource(R.string.ui_version, BuildConfig.VERSION_NAME), color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(stringResource(R.string.about_summary))
        TextButton(onClick = { help = true }) { Text(stringResource(R.string.about_help)) }
        TextButton(onClick = { uri.openUri("https://github.com/lelloman/simple-ai/issues") }) { Text(stringResource(R.string.ui_support_and_issue_tracker)) }
        TextButton(onClick = {
            context.getSystemService(android.content.ClipboardManager::class.java).setPrimaryClip(
                android.content.ClipData.newPlainText("SimpleAI diagnostics", supportDiagnostics()))
            android.widget.Toast.makeText(context, context.getString(R.string.ui_diagnostics_copied), android.widget.Toast.LENGTH_SHORT).show()
        }) { Text(stringResource(R.string.ui_copy_support_diagnostics)) }
        TextButton(onClick = { licenses = true }) { Text(stringResource(R.string.ui_model_and_library_licenses)) }
    }
    if (help) AlertDialog(onDismissRequest = { help = false }, title = { Text(stringResource(R.string.about_help)) }, text = {
        Text(stringResource(R.string.about_help_body))
    }, confirmButton = { TextButton(onClick = { help = false }) { Text(stringResource(R.string.ui_close)) } })
    if (licenses) AlertDialog(onDismissRequest = { licenses = false }, title = { Text(stringResource(R.string.ui_model_and_library_licenses)) }, text = {
        Text(stringResource(R.string.about_licenses))
    }, confirmButton = { TextButton(onClick = { licenses = false }) { Text(stringResource(R.string.ui_close)) } })
}
