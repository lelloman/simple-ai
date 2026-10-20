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
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.lelloman.lellodesign.LelloAlert
import com.lelloman.lellodesign.LelloAvatar
import com.lelloman.lellodesign.LelloButton
import com.lelloman.lellodesign.LelloCard
import com.lelloman.lellodesign.LelloOutlinedButton
import com.lelloman.lellodesign.LelloSection
import com.lelloman.lellodesign.LelloTextButton
import com.lelloman.lellodesign.LelloTone
import com.lelloman.simpleai.R
import com.lelloman.simpleai.cloud.CloudEndpoint
import com.lelloman.simpleai.cloud.GatewayLoginActivity
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen(viewModel: CapabilitiesViewModel, onAbout: () -> Unit) {
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
    val openServerDialog = { draft = endpoint; saveFailed = false; cloudInfo = true }
    val openSignIn = { context.startActivity(Intent(context, GatewayLoginActivity::class.java)) }
    SimplePage(stringResource(R.string.nav_settings)) {
        LelloSection(stringResource(R.string.settings_downloads)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.settings_mobile), style = MaterialTheme.typography.titleMedium)
                    Text(stringResource(R.string.settings_mobile_hint), style = MaterialTheme.typography.bodySmall)
                }
                Switch(state.allowMeteredDownloads, viewModel::setAllowMeteredDownloads,
                    modifier = Modifier.semantics { contentDescription = context.getString(R.string.settings_mobile) })
            }
        }
        state.storage?.let {
            LelloSection(stringResource(R.string.settings_storage)) {
                Text(stringResource(R.string.settings_storage_values, formatSize(it.usedBytes), formatSize(it.availableBytes)),
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        LelloSection(stringResource(R.string.settings_account)) {
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
        }
        LelloSection(stringResource(R.string.ui_cloud_ai)) {
            ModelRow(stringResource(R.string.cloud_server_title), endpoint.ifBlank { stringResource(R.string.settings_not_configured) },
                openServerDialog)
            Text(stringResource(R.string.cloud_server_auth), style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        state.serviceError?.let {
            Text(it, color = MaterialTheme.colorScheme.error)
            LelloTextButton(onClick = viewModel::refreshCapabilities) { Text(stringResource(R.string.ui_retry_connection)) }
        }
        ModelRow(stringResource(R.string.settings_about), stringResource(R.string.settings_about_hint), onAbout)
    }
    if (cloudInfo) {
        val valid = draft.isBlank() || CloudEndpoint.chatUrl(draft) != null
        AlertDialog(
            onDismissRequest = { if (!saving) cloudInfo = false },
            title = { Text(stringResource(R.string.cloud_server_title)) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = draft, onValueChange = { draft = it; saveFailed = false },
                        label = { Text(stringResource(R.string.cloud_server_url)) },
                        placeholder = { Text("https://ai.example.com") },
                        singleLine = true, enabled = !saving, isError = !valid,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                        supportingText = { Text(stringResource(if (valid) R.string.cloud_server_hint else R.string.cloud_server_invalid)) }
                    )
                    Text(stringResource(R.string.cloud_server_auth), style = MaterialTheme.typography.bodySmall)
                    if (saveFailed) Text(stringResource(R.string.cloud_server_save_failed), color = MaterialTheme.colorScheme.error)
                }
            },
            confirmButton = {
                LelloTextButton(enabled = valid && !saving, onClick = {
                    saving = true
                    scope.launch {
                        try {
                            if (viewModel.saveCloudEndpoint(draft)) cloudInfo = false else saveFailed = true
                        } finally { saving = false }
                    }
                }) { Text(stringResource(R.string.cloud_server_save)) }
            },
            dismissButton = { LelloTextButton(enabled = !saving, onClick = { cloudInfo = false }) { Text(stringResource(R.string.cloud_server_cancel)) } }
        )
    }
}

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
    LelloCard {
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
private fun SignedOutAccount(server: String, onSignIn: () -> Unit, onConfigureServer: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        LelloAvatar("")
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(stringResource(if (server.isBlank()) R.string.gateway_no_server_title else R.string.gateway_status_signed_out),
                style = MaterialTheme.typography.titleMedium)
            Text(stringResource(if (server.isBlank()) R.string.gateway_no_server_hint else R.string.gateway_signed_out_hint),
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
    if (server.isNotBlank()) {
        Text(stringResource(R.string.gateway_server_line, serverName(server)), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    if (server.isBlank()) {
        LelloButton(modifier = Modifier.fillMaxWidth(), onClick = onConfigureServer) { Text(stringResource(R.string.gateway_configure_server)) }
    } else {
        LelloButton(modifier = Modifier.fillMaxWidth(), onClick = onSignIn) { Text(stringResource(R.string.gateway_sign_in)) }
    }
}

@Composable
private fun SigningInAccount(server: String, onSignIn: () -> Unit, onCancelSignIn: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        CircularProgressIndicator(Modifier.size(36.dp), strokeWidth = 3.dp)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(stringResource(R.string.gateway_signing_in), style = MaterialTheme.typography.titleMedium)
            Text(stringResource(R.string.gateway_signing_in_hint), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
    Text(stringResource(R.string.gateway_server_line, serverName(server)), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    LelloButton(modifier = Modifier.fillMaxWidth(), onClick = onSignIn) { Text(stringResource(R.string.gateway_sign_in_again)) }
    LelloTextButton(modifier = Modifier.fillMaxWidth(), onClick = onCancelSignIn) { Text(stringResource(R.string.gateway_cancel_sign_in)) }
}

@Composable
private fun SignedInAccount(username: String?, server: String, onSignIn: () -> Unit, onSignOut: () -> Unit) {
    Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        LelloAvatar(username.orEmpty())
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(username ?: stringResource(R.string.gateway_profile_unavailable), style = MaterialTheme.typography.titleMedium,
                maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(stringResource(R.string.gateway_status_signed_in), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
    Text(stringResource(R.string.gateway_server_line, serverName(server)), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    if (username == null) {
        Text(stringResource(R.string.gateway_profile_unavailable_hint), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        LelloButton(modifier = Modifier.fillMaxWidth(), onClick = onSignIn) { Text(stringResource(R.string.gateway_sign_in_again)) }
    }
    LelloOutlinedButton(modifier = Modifier.fillMaxWidth(), onClick = onSignOut) { Text(stringResource(R.string.gateway_sign_out)) }
}

private fun serverName(endpoint: String): String {
    val trimmed = endpoint.trim()
    if (trimmed.isEmpty()) return trimmed
    return runCatching { android.net.Uri.parse(trimmed).host }.getOrNull()?.takeIf { it.isNotBlank() } ?: trimmed
}
