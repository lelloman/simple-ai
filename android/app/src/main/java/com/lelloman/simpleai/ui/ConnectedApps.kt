package com.lelloman.simpleai.ui

import android.content.Context
import android.content.pm.PackageManager
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.core.graphics.drawable.toBitmap
import com.lelloman.lellodesign.LelloButton
import com.lelloman.lellodesign.LelloDimensions
import com.lelloman.lellodesign.LelloIllustration
import com.lelloman.lellodesign.LelloIllustrationKind
import com.lelloman.lellodesign.LelloTextButton
import com.lelloman.lellodesign.LelloTextStyles
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.text.font.FontWeight
import com.lelloman.lellodesign.LelloState
import com.lelloman.lellodesign.LelloWorkspace
import com.lelloman.lellodesign.LocalLelloPalette
import com.lelloman.simpleai.R
import com.lelloman.simpleai.access.ClientAccess
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

@Composable
fun ConnectedApps() {
    val context = LocalContext.current
    val access = remember(context) { ClientAccess.get(context) }
    val clients by access.clients.collectAsState()
    ConnectedAppsContent(clients = clients, onSetApproved = access::setApproved)
}

@Composable
internal fun ConnectedAppsContent(
    clients: List<ClientAccess.Client>,
    onSetApproved: (ClientAccess.Client, Boolean) -> Unit,
    metadata: @Composable (List<String>) -> AppMetadata = { rememberAppMetadata(it) },
) {
    val ordered = remember(clients) { orderConnectedApps(clients) }

    if (ordered.isEmpty()) {
        LelloWorkspace(Modifier.fillMaxSize()) {
            LelloState(
                title = stringResource(R.string.apps_empty_title),
                description = stringResource(R.string.apps_empty_hint),
                modifier = Modifier.fillMaxWidth(),
                illustration = { LelloIllustration(LelloIllustrationKind.Empty) },
            )
        }
    } else {
        LelloWorkspace(Modifier.fillMaxSize()) {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(bottom = LelloDimensions.space6),
            ) {
                item {
                    Text(
                        stringResource(R.string.apps_hint),
                        modifier = Modifier.padding(bottom = LelloDimensions.space2),
                        style = MaterialTheme.typography.bodyLarge,
                        color = LocalLelloPalette.current["text-secondary"],
                    )
                }
                items(ordered, key = { it.identity }) { client ->
                    Column {
                        ConnectedAppRow(client, onSetApproved = onSetApproved, metadata = metadata)
                        RowDivider()
                    }
                }
            }
        }
    }
}

@Composable
private fun ConnectedAppRow(
    client: ClientAccess.Client,
    onSetApproved: (ClientAccess.Client, Boolean) -> Unit,
    metadata: @Composable (List<String>) -> AppMetadata,
) {
    val packageNames = remember(client.packages) { parsePackageNames(client.packages) }
    val appMetadata = metadata(packageNames)
    val fallbackName = stringResource(R.string.apps_unknown_name)
    val displayName = appMetadata.label ?: fallbackName
    val actionTarget = appMetadata.label ?: client.packages
    val approveLabel = stringResource(R.string.apps_approve_label, actionTarget)
    val revokeLabel = stringResource(R.string.apps_revoke_label, actionTarget)

    Column(
        modifier = Modifier.fillMaxWidth().padding(vertical = LelloDimensions.space4),
        verticalArrangement = Arrangement.spacedBy(LelloDimensions.space2),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space4),
            verticalAlignment = Alignment.Top,
        ) {
            AppIcon(appMetadata.icon)
            Column(
                modifier = Modifier
                    .weight(1f)
                    .semantics(mergeDescendants = true) {},
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(displayName, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                Text(
                    packageNames.joinToString(", "),
                    style = LelloTextStyles.caption,
                )
                Box(Modifier.padding(top = LelloDimensions.space1)) {
                    StatusPill(
                        stringResource(if (client.approved) R.string.ui_approved else R.string.apps_waiting),
                        if (client.approved) PillTone.Success else PillTone.Warning,
                    )
                }
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            if (client.approved) {
                LelloTextButton(
                    onClick = { onSetApproved(client, false) },
                    modifier = Modifier.semantics { contentDescription = revokeLabel },
                ) {
                    Text(stringResource(R.string.apps_revoke))
                }
            } else {
                LelloButton(
                    onClick = { onSetApproved(client, true) },
                    modifier = Modifier.semantics { contentDescription = approveLabel },
                ) {
                    Text(stringResource(R.string.apps_approve))
                }
            }
        }
    }
}

@Composable
private fun AppIcon(icon: ImageBitmap?) {
    Surface(
        modifier = Modifier.size(40.dp),
        shape = MaterialTheme.shapes.small,
        color = LocalLelloPalette.current["neutral"],
    ) {
        if (icon != null) {
            Image(
                bitmap = icon,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        } else {
            Icon(
                painterResource(R.drawable.ic_app_fallback),
                contentDescription = null,
                modifier = Modifier.fillMaxSize().padding(10.dp),
                tint = LocalLelloPalette.current["text-secondary"],
            )
        }
    }
}

internal data class AppMetadata(val label: String?, val icon: ImageBitmap?)

@Composable
private fun rememberAppMetadata(packageNames: List<String>): AppMetadata {
    val context = LocalContext.current
    val metadata by produceState(AppMetadata(null, null), packageNames) {
        value = withContext(Dispatchers.IO) { loadAppMetadata(context, packageNames) }
    }
    return metadata
}

private fun loadAppMetadata(context: Context, packageNames: List<String>): AppMetadata {
    val packageManager = context.packageManager
    val iconSize = (48 * context.resources.displayMetrics.density).roundToInt().coerceAtLeast(1)
    for (packageName in packageNames) {
        val info = try {
            packageManager.getApplicationInfo(packageName, 0)
        } catch (_: PackageManager.NameNotFoundException) {
            continue
        } catch (_: SecurityException) {
            continue
        }
        val label = try {
            packageManager.getApplicationLabel(info).toString().trim().takeIf { it.isNotEmpty() }
        } catch (_: Exception) {
            null
        }
        val icon = try {
            packageManager.getApplicationIcon(info).toBitmap(iconSize, iconSize).asImageBitmap()
        } catch (_: Exception) {
            null
        }
        return AppMetadata(label, icon)
    }
    return AppMetadata(null, null)
}

internal fun parsePackageNames(packages: String): List<String> =
    packages.split(',').map { it.trim() }.filter { it.isNotEmpty() }.distinct()

internal fun orderConnectedApps(clients: List<ClientAccess.Client>): List<ClientAccess.Client> =
    clients.sortedWith(compareBy({ it.approved }, { it.packages }))
