package com.lelloman.simpleai.ui

import android.content.Intent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.lelloman.lellodesign.LelloAlert
import com.lelloman.lellodesign.LelloAppearance
import com.lelloman.lellodesign.LelloAppearanceSelector
import com.lelloman.lellodesign.LelloAvatar
import com.lelloman.lellodesign.LelloButton
import com.lelloman.lellodesign.LelloDimensions
import com.lelloman.lellodesign.LelloSettingsSection
import com.lelloman.lellodesign.LelloTextButton
import com.lelloman.lellodesign.LelloTextStyles
import com.lelloman.lellodesign.LelloTone
import com.lelloman.lellodesign.LocalLelloPalette
import com.lelloman.simpleai.R
import com.lelloman.simpleai.cloud.CloudEndpoint
import com.lelloman.simpleai.cloud.GatewayLoginActivity
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen(
    viewModel: CapabilitiesViewModel,
    appearance: LelloAppearance,
    onAppearance: (LelloAppearance) -> Unit,
    onAbout: () -> Unit,
) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsState()
    val signedIn by viewModel.gatewaySignedIn.collectAsState()
    val signingIn by viewModel.gatewaySigningIn.collectAsState()
    val account by viewModel.gatewayAccount.collectAsState()
    val authError by viewModel.gatewayAuthError.collectAsState()
    val endpoint by viewModel.cloudEndpoint.collectAsState()
    var cloudInfo by rememberSaveable { mutableStateOf(false) }
    var draft by rememberSaveable { mutableStateOf("") }
    var saveFailed by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val secondary = LocalLelloPalette.current["text-secondary"]
    val openServerDialog = { draft = endpoint; saveFailed = false; cloudInfo = true }
    val openSignIn = { context.startActivity(Intent(context, GatewayLoginActivity::class.java)) }
    SimplePage {
        state.serviceError?.let {
            LelloAlert(stringResource(R.string.settings_service_problem), tone = LelloTone.Error) {
                Text(it)
                LelloTextButton(onClick = viewModel::refreshCapabilities) { Text(stringResource(R.string.ui_retry_connection)) }
            }
        }
        LelloSettingsSection(stringResource(R.string.ui_cloud_ai)) {
            GatewayAccountCard(
                signedIn = signedIn,
                signingIn = signingIn,
                username = account?.username,
                server = endpoint,
                authError = authError,
                onSignIn = openSignIn,
                onSignOut = viewModel::signOutGateway,
                onCancelSignIn = viewModel::cancelGatewaySignIn,
                onConfigureServer = openServerDialog,
            )
            Column {
                RowDivider()
                NavigationRow(stringResource(R.string.cloud_server_title),
                    endpoint.ifBlank { stringResource(R.string.settings_not_configured) }, openServerDialog)
                RowDivider()
                Text(stringResource(R.string.cloud_server_auth), Modifier.padding(top = LelloDimensions.space3),
                    style = LelloTextStyles.caption)
            }
        }
        LelloSettingsSection(stringResource(R.string.settings_downloads)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space4)) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(stringResource(R.string.settings_mobile), style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                    Text(stringResource(R.string.settings_mobile_hint), style = MaterialTheme.typography.bodyMedium, color = secondary)
                }
                Switch(state.allowMeteredDownloads, viewModel::setAllowMeteredDownloads,
                    modifier = Modifier.semantics { contentDescription = context.getString(R.string.settings_mobile) })
            }
            state.storage?.let {
                Column {
                    RowDivider()
                    DetailRow(stringResource(R.string.settings_storage)) {
                        Text(stringResource(R.string.settings_storage_values, formatSize(it.usedBytes), formatSize(it.availableBytes)),
                            style = LelloTextStyles.timestamp)
                    }
                }
            }
        }
        LelloSettingsSection(stringResource(R.string.settings_appearance)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space4)) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(stringResource(R.string.settings_theme), style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                    Text(stringResource(when (appearance) {
                        LelloAppearance.Light -> R.string.settings_theme_light
                        LelloAppearance.Dark -> R.string.settings_theme_dark
                        LelloAppearance.System -> R.string.settings_theme_system
                    }), style = MaterialTheme.typography.bodyMedium, color = secondary)
                }
                LelloAppearanceSelector(appearance, onAppearance)
            }
        }
        LelloSettingsSection(stringResource(R.string.settings_help)) {
            NavigationRow(stringResource(R.string.settings_about), stringResource(R.string.settings_about_hint), onAbout)
        }
    }
    if (cloudInfo) {
        val valid = draft.isBlank() || CloudEndpoint.chatUrl(draft) != null
        CalmDialog(
            title = stringResource(R.string.cloud_server_title),
            onDismiss = { if (!saving) cloudInfo = false },
            confirmLabel = stringResource(R.string.cloud_server_save),
            confirmEnabled = valid && !saving,
            onConfirm = {
                saving = true
                scope.launch {
                    try {
                        if (viewModel.saveCloudEndpoint(draft)) cloudInfo = false else saveFailed = true
                    } finally { saving = false }
                }
            },
            dismissLabel = stringResource(R.string.cloud_server_cancel),
            dismissEnabled = !saving,
        ) {
            OutlinedTextField(
                value = draft, onValueChange = { draft = it; saveFailed = false },
                label = { Text(stringResource(R.string.cloud_server_url)) },
                placeholder = { Text("https://ai.example.com") },
                singleLine = true, enabled = !saving, isError = !valid,
                shape = MaterialTheme.shapes.small,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                supportingText = { Text(stringResource(if (valid) R.string.cloud_server_hint else R.string.cloud_server_invalid)) }
            )
            Text(stringResource(R.string.cloud_server_auth), style = LelloTextStyles.caption)
            if (saveFailed) Text(stringResource(R.string.cloud_server_save_failed), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.error)
        }
    }
}

/** Open account block for the gateway sign-in; no enclosing card. */
@Composable
internal fun GatewayAccountCard(
    signedIn: Boolean,
    signingIn: Boolean,
    username: String?,
    server: String,
    authError: String?,
    onSignIn: () -> Unit,
    onSignOut: () -> Unit,
    onCancelSignIn: () -> Unit,
    onConfigureServer: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(LelloDimensions.space4)) {
        when {
            signingIn -> SigningInAccount(server, onSignIn, onCancelSignIn)
            signedIn -> SignedInAccount(username, server, onSignIn, onSignOut)
            else -> SignedOutAccount(server, onSignIn, onConfigureServer)
        }
        if (authError != null && !signingIn && (!signedIn || username == null)) {
            LelloAlert(title = stringResource(R.string.gateway_auth_error), tone = LelloTone.Error) {
                Text(authError)
            }
        }
    }
}

@Composable
private fun AccountHeader(leading: @Composable () -> Unit, title: String, detail: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space4)) {
        leading()
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium,
                maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(detail, style = MaterialTheme.typography.bodyMedium, color = LocalLelloPalette.current["text-secondary"])
        }
    }
}

@Composable
private fun SignedOutAccount(server: String, onSignIn: () -> Unit, onConfigureServer: () -> Unit) {
    AccountHeader(
        { LelloAvatar("") },
        stringResource(if (server.isBlank()) R.string.gateway_no_server_title else R.string.gateway_status_signed_out),
        if (server.isBlank()) stringResource(R.string.gateway_no_server_hint)
        else stringResource(R.string.gateway_signed_out_hint) + "\n" + stringResource(R.string.gateway_server_line, serverName(server)),
    )
    if (server.isBlank()) {
        LelloButton(onClick = onConfigureServer) { Text(stringResource(R.string.gateway_configure_server)) }
    } else {
        LelloButton(onClick = onSignIn) { Text(stringResource(R.string.gateway_sign_in)) }
    }
}

@Composable
private fun SigningInAccount(server: String, onSignIn: () -> Unit, onCancelSignIn: () -> Unit) {
    AccountHeader(
        { Box(Modifier.size(LelloDimensions.avatarSize), contentAlignment = Alignment.Center) {
            CircularProgressIndicator(Modifier.size(28.dp), strokeWidth = 3.dp)
        } },
        stringResource(R.string.gateway_signing_in),
        stringResource(R.string.gateway_signing_in_hint) + "\n" + stringResource(R.string.gateway_server_line, serverName(server)),
    )
    Row(horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space2)) {
        LelloButton(onClick = onSignIn) { Text(stringResource(R.string.gateway_sign_in_again)) }
        LelloTextButton(onClick = onCancelSignIn) { Text(stringResource(R.string.gateway_cancel_sign_in)) }
    }
}

@Composable
private fun SignedInAccount(username: String?, server: String, onSignIn: () -> Unit, onSignOut: () -> Unit) {
    AccountHeader(
        { LelloAvatar(username.orEmpty()) },
        username ?: stringResource(R.string.gateway_profile_unavailable),
        stringResource(R.string.gateway_status_signed_in_server, serverName(server)),
    )
    if (username == null) {
        Text(stringResource(R.string.gateway_profile_unavailable_hint), style = MaterialTheme.typography.bodyMedium,
            color = LocalLelloPalette.current["text-secondary"])
    }
    Row(horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space2)) {
        if (username == null) LelloButton(onClick = onSignIn) { Text(stringResource(R.string.gateway_sign_in_again)) }
        LelloTextButton(onClick = onSignOut) { Text(stringResource(R.string.gateway_sign_out)) }
    }
}

private fun serverName(endpoint: String): String {
    val trimmed = endpoint.trim()
    if (trimmed.isEmpty()) return trimmed
    return runCatching { android.net.Uri.parse(trimmed).host }.getOrNull()?.takeIf { it.isNotBlank() } ?: trimmed
}
