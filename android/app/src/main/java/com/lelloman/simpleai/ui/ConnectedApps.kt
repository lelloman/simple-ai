package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.lelloman.simpleai.R
import com.lelloman.simpleai.access.ClientAccess

@Composable
fun ConnectedApps() {
    val context = LocalContext.current
    val access = remember(context) { ClientAccess.get(context) }
    val clients by access.clients.collectAsState()
    SimplePage(stringResource(R.string.nav_apps)) {
        if (clients.isEmpty()) {
            Text(stringResource(R.string.apps_empty_title), style = MaterialTheme.typography.titleLarge)
            Text(stringResource(R.string.apps_empty_hint), color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            Text(stringResource(R.string.apps_hint), style = MaterialTheme.typography.bodyMedium)
            clients.sortedBy { it.approved }.forEach { client ->
                OutlinedCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(client.packages, style = MaterialTheme.typography.titleMedium)
                        Text(stringResource(if (client.approved) R.string.ui_approved else R.string.apps_waiting), color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (client.approved) TextButton(onClick = { access.setApproved(client, false) }, modifier = Modifier.semantics { contentDescription = context.getString(R.string.apps_revoke_label, client.packages) }) { Text(stringResource(R.string.apps_revoke)) }
                        else Button(onClick = { access.setApproved(client, true) }, modifier = Modifier.semantics { contentDescription = context.getString(R.string.apps_approve_label, client.packages) }) { Text(stringResource(R.string.apps_approve)) }
                    }
                }
            }
        }
    }
}
