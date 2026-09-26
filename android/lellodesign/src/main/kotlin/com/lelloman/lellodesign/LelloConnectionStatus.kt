package com.lelloman.lellodesign

import androidx.compose.foundation.background
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.input.key.*
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.*
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupPositionProvider
import androidx.compose.ui.window.PopupProperties

enum class LelloConnectionState { Connected, Connecting, Disconnected }

data class LelloConnectionLabels(
    val title: String = "Connection status",
    val connected: String = "Connected",
    val connecting: String = "Connecting…",
    val disconnected: String = "Disconnected",
) {
    fun label(state: LelloConnectionState): String = when (state) {
        LelloConnectionState.Connected -> connected
        LelloConnectionState.Connecting -> connecting
        LelloConnectionState.Disconnected -> disconnected
    }
}

/** Presentation only: the host supplies connectivity and localized labels.
 * A native 48 dp target contains the static 10 dp semantic dot. The tooltip
 * retains anchor focus and opens on tap, keyboard focus or mouse hover.
 */
@Composable
fun LelloConnectionStatus(
    state: LelloConnectionState,
    modifier: Modifier = Modifier,
    labels: LelloConnectionLabels = LelloConnectionLabels(),
) {
    val palette = LocalLelloPalette.current
    val text = labels.label(state)
    val color = palette[when (state) {
        LelloConnectionState.Connected -> "success"
        LelloConnectionState.Connecting -> "warning"
        LelloConnectionState.Disconnected -> "error"
    }]
    val interactions = remember { MutableInteractionSource() }
    val hovered by interactions.collectIsHoveredAsState()
    var focused by remember { mutableStateOf(false) }
    var pinned by remember { mutableStateOf(false) }
    var dismissed by remember { mutableStateOf(false) }
    LaunchedEffect(hovered) { if (hovered) dismissed = false }
    val visible = !dismissed && (hovered || focused || pinned)
    val dismiss = { pinned = false; dismissed = true }
    Box(modifier) {
        IconButton(
            onClick = { pinned = !pinned; dismissed = !pinned },
            interactionSource = interactions,
            modifier = Modifier.size(48.dp)
                .onFocusChanged {
                    if (focused && !it.isFocused) dismissed = true
                    focused = it.isFocused
                    if (it.isFocused) dismissed = false else pinned = false
                }
                .hoverable(interactions)
                .onPreviewKeyEvent {
                    if (it.key == Key.Escape && visible) {
                        if (it.type == KeyEventType.KeyDown) dismiss()
                        true
                    } else false
                }
                .semantics {
                    contentDescription = labels.title
                    stateDescription = text
                    liveRegion = LiveRegionMode.Polite
                },
        ) {
            Box(Modifier.size(10.dp).background(color, CircleShape))
        }
        if (visible) {
            Popup(
                popupPositionProvider = remember { ConnectionTooltipPosition() },
                onDismissRequest = dismiss,
                properties = PopupProperties(focusable = false, dismissOnClickOutside = true),
            ) {
                Surface(shape = RoundedCornerShape(4.dp),
                    color = MaterialTheme.colorScheme.inverseSurface,
                    contentColor = MaterialTheme.colorScheme.inverseOnSurface) {
                    // The anchor already announces status; avoid duplicate accessibility output.
                    Text(text, Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                        .semantics { hideFromAccessibility() }, style = MaterialTheme.typography.bodySmall.copy(fontSize = 12.sp))
                }
            }
        }
    }
}

/** End-align to the target and keep the label inside narrow windows in either direction. */
private class ConnectionTooltipPosition : PopupPositionProvider {
    override fun calculatePosition(anchorBounds: IntRect, windowSize: IntSize,
        layoutDirection: LayoutDirection, popupContentSize: IntSize): IntOffset {
        val x = if (layoutDirection == LayoutDirection.Ltr) anchorBounds.right - popupContentSize.width else anchorBounds.left
        val y = if (anchorBounds.bottom + popupContentSize.height <= windowSize.height) anchorBounds.bottom
            else anchorBounds.top - popupContentSize.height
        return IntOffset(x.coerceIn(0, (windowSize.width - popupContentSize.width).coerceAtLeast(0)),
            y.coerceIn(0, (windowSize.height - popupContentSize.height).coerceAtLeast(0)))
    }
}
