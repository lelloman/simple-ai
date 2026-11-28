package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.lelloman.lellodesign.LelloSection
import com.lelloman.lellodesign.LelloTextStyles
import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.R

@Composable
fun AboutScreen(@Suppress("UNUSED_PARAMETER") onBack: () -> Unit) {
    val context = LocalContext.current
    val uri = LocalUriHandler.current
    var help by remember { mutableStateOf(false) }
    var licenses by remember { mutableStateOf(false) }
    SimplePage {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(stringResource(R.string.ui_simpleai), style = MaterialTheme.typography.titleLarge)
            Text(stringResource(R.string.ui_version, BuildConfig.VERSION_NAME), style = LelloTextStyles.caption)
        }
        Text(stringResource(R.string.about_summary), style = MaterialTheme.typography.bodyLarge)
        LelloSection(stringResource(R.string.about_help_and_support)) {
            Column {
                NavigationRow(stringResource(R.string.about_help), null, { help = true }, trailingIcon = R.drawable.ic_help)
                RowDivider()
                NavigationRow(stringResource(R.string.ui_support_and_issue_tracker), null,
                    { uri.openUri("https://github.com/lelloman/simple-ai/issues") }, trailingIcon = R.drawable.ic_open_in_new)
                RowDivider()
                NavigationRow(stringResource(R.string.ui_copy_support_diagnostics), stringResource(R.string.about_diagnostics_hint), {
                    context.getSystemService(android.content.ClipboardManager::class.java).setPrimaryClip(
                        android.content.ClipData.newPlainText("SimpleAI diagnostics", supportDiagnostics()))
                    android.widget.Toast.makeText(context, context.getString(R.string.ui_diagnostics_copied), android.widget.Toast.LENGTH_SHORT).show()
                }, trailingIcon = R.drawable.ic_copy)
                RowDivider()
                NavigationRow(stringResource(R.string.ui_model_and_library_licenses), null, { licenses = true },
                    trailingIcon = R.drawable.ic_license)
            }
        }
    }
    if (help) CalmDialog(stringResource(R.string.about_help), { help = false }, stringResource(R.string.ui_close), { help = false }) {
        Text(stringResource(R.string.about_help_body))
    }
    if (licenses) CalmDialog(stringResource(R.string.ui_model_and_library_licenses), { licenses = false },
        stringResource(R.string.ui_close), { licenses = false }) {
        Text(stringResource(R.string.about_licenses))
    }
}
