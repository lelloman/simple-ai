package com.lelloman.simpleai.ui

import com.lelloman.simpleai.R

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import com.lelloman.simpleai.access.ClientAccess

@Composable
fun ConnectedApps() {
    val strings = androidx.compose.ui.platform.LocalContext.current
    val context = LocalContext.current
    val access = remember(context) { ClientAccess.get(context) }
    val clients by access.clients.collectAsState()
    var expanded by remember { mutableStateOf(false) }
    TextButton(onClick = { expanded = !expanded }) { Text(strings.getString(R.string.ui_connected_apps_approved, clients.count { it.approved })) }
    if (expanded) {
        Text(strings.getString(R.string.ui_approve_apps_you_trust_to_use_downloaded_models_and_cloud_request))
        if (clients.isEmpty()) Text(strings.getString(R.string.ui_no_apps_have_requested_access_connect_from_a_compatible_app_then_))
        clients.forEach { client ->
            Column {
                Text(client.packages, style = MaterialTheme.typography.titleSmall)
                Text(if (client.approved) strings.getString(R.string.ui_approved) else strings.getString(R.string.ui_not_approved))
                TextButton(onClick = { access.setApproved(client, !client.approved) }) {
                    Text(if (client.approved) strings.getString(R.string.ui_revoke, client.packages) else strings.getString(R.string.ui_approve, client.packages))
                }
            }
        }
    }
}
