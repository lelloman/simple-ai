package com.lelloman.simpleai.ui

import android.content.Context
import android.content.pm.PackageManager
import androidx.compose.foundation.BorderStroke
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
import com.lelloman.lellodesign.LelloOutlinedButton
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
    val ordered = remember(clients) { orderConnectedApps(clients) }

    if (ordered.isEmpty()) {
        LelloWorkspace(Modifier.fillMaxSize()) {
            LelloState(
                title = stringResource(R.string.apps_empty_title),
                description = stringResource(R.string.apps_empty_hint),
                modifier = Modifier.fillMaxWidth(),
                icon = {
                    Icon(
                        painterResource(R.drawable.ic_app_fallback),
                        contentDescription = null,
                        modifier = Modifier.size(LelloDimensions.space8),
                    )
                },
            )
        }
    } else {
        LelloWorkspace(Modifier.fillMaxSize()) {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(bottom = LelloDimensions.space6),
                verticalArrangement = Arrangement.spacedBy(LelloDimensions.space3),
            ) {
                item {
                    Text(
                        stringResource(R.string.apps_hint),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                items(ordered, key = { it.identity }) { client ->
                    ConnectedAppRow(client, onSetApproved = access::setApproved)
                }
            }
        }
    }
}

@Composable
private fun ConnectedAppRow(
    client: ClientAccess.Client,
    onSetApproved: (ClientAccess.Client, Boolean) -> Unit,
) {
    val packageNames = remember(client.packages) { parsePackageNames(client.packages) }
    val metadata = rememberAppMetadata(packageNames)
    val fallbackName = stringResource(R.string.apps_unknown_name)
    val displayName = metadata.label ?: fallbackName
    val actionTarget = metadata.label ?: client.packages
    val approveLabel = stringResource(R.string.apps_approve_label, actionTarget)
    val revokeLabel = stringResource(R.string.apps_revoke_label, actionTarget)

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surface,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
    ) {
        Column(
            modifier = Modifier.padding(LelloDimensions.space4),
            verticalArrangement = Arrangement.spacedBy(LelloDimensions.space3),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space3),
                verticalAlignment = Alignment.Top,
            ) {
                AppIcon(metadata.icon)
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .semantics(mergeDescendants = true) {},
                    verticalArrangement = Arrangement.spacedBy(LelloDimensions.space1),
                ) {
                    Text(displayName, style = MaterialTheme.typography.titleMedium)
                    Text(
                        packageNames.joinToString(", "),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    ApprovalTag(client.approved)
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                if (client.approved) {
                    LelloOutlinedButton(
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
}

@Composable
private fun AppIcon(icon: ImageBitmap?) {
    Surface(
        modifier = Modifier.size(48.dp),
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.surfaceVariant,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
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
                modifier = Modifier.fillMaxSize().padding(LelloDimensions.space3),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ApprovalTag(approved: Boolean) {
    val palette = LocalLelloPalette.current
    val role = if (approved) "success" else "warning"
    Surface(
        shape = MaterialTheme.shapes.small,
        color = palette["$role-container"],
        contentColor = palette["on-$role-container"],
        border = BorderStroke(1.dp, palette["$role-border"]),
    ) {
        Text(
            stringResource(if (approved) R.string.ui_approved else R.string.apps_waiting),
            modifier = Modifier.padding(horizontal = LelloDimensions.space2, vertical = 2.dp),
            style = MaterialTheme.typography.labelMedium,
        )
    }
}

private data class AppMetadata(val label: String?, val icon: ImageBitmap?)

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
