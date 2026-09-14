package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import com.lelloman.simpleai.access.ClientAccess

@Composable
fun ConnectedApps() {
    val context = LocalContext.current
    val access = remember(context) { ClientAccess.get(context) }
    val clients by access.clients.collectAsState()
    var expanded by remember { mutableStateOf(false) }
    TextButton(onClick = { expanded = !expanded }) { Text("Connected apps (${clients.count { it.approved }} approved)") }
    if (expanded) {
        Text("Approve apps you trust to use downloaded models and cloud requests. Revoking access blocks new requests; an active request can finish.")
        if (clients.isEmpty()) Text("No apps have requested access. Connect from a compatible app, then return here to approve it.")
        clients.forEach { client ->
            Column {
                Text(client.packages, style = MaterialTheme.typography.titleSmall)
                Text(if (client.approved) "Approved" else "Not approved")
                TextButton(onClick = { access.setApproved(client, !client.approved) }) {
                    Text("${if (client.approved) "Revoke" else "Approve"} ${client.packages}")
                }
            }
        }
    }
}
