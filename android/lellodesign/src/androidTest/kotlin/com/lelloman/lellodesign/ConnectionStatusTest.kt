package com.lelloman.lellodesign

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toPixelMap
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.semantics.*
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.*
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

@OptIn(ExperimentalTestApi::class)
class ConnectionStatusTest {
    @get:Rule val compose = createComposeRule()

    @Test fun statusIsLocalizedAndUsesSemanticColorsAcrossThemes() {
        var status by mutableStateOf(LelloConnectionState.Connected)
        var dark by mutableStateOf(false)
        val labels = LelloConnectionLabels("Home Assistant", "Connesso", "Connessione…", "Disconnesso")
        compose.setContent { LelloTheme("blue", dark) { LelloConnectionStatus(status, labels = labels) } }
        val control = compose.onNodeWithContentDescription("Home Assistant")
        control.assertHeightIsEqualTo(48.dp).assertWidthIsEqualTo(48.dp)
        control.assert(SemanticsMatcher.expectValue(SemanticsProperties.LiveRegion, LiveRegionMode.Polite))
        for (night in listOf(false, true)) for (state in LelloConnectionState.entries) {
            compose.runOnIdle { dark = night; status = state }
            control.assert(SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, labels.label(state)))
            val token = when (state) {
                LelloConnectionState.Connected -> "success"
                LelloConnectionState.Connecting -> "warning"
                LelloConnectionState.Disconnected -> "error"
            }
            val pixels = control.captureToImage().toPixelMap()
            assertEquals(LelloPalettes.forProduct("blue", night)[token], pixels[pixels.width / 2, pixels.height / 2])
        }
    }

    @Test fun tapToggleAndOutsideTapDismissWithoutLosingDraft() {
        lateinit var hostView: android.view.View
        compose.setContent { LelloTheme {
            hostView = androidx.compose.ui.platform.LocalView.current
            Column {
                LelloConnectionStatus(LelloConnectionState.Connected)
                var draft by remember { mutableStateOf("") }
                LelloTextField(draft, { draft = it }, { Text("Draft") })
            }
        } }
        compose.onNodeWithText("Draft").performTextInput("Keep me")
        val control = compose.onNodeWithContentDescription("Connection status")
        control.performClick()
        compose.onNode(isPopup()).assertExists()
        control.performClick()
        compose.onNode(isPopup()).assertDoesNotExist()
        control.performClick()
        val point = compose.onNodeWithText("Keep me").fetchSemanticsNode().boundsInRoot.center
        val location = IntArray(2)
        compose.runOnIdle { hostView.getLocationOnScreen(location) }
        val automation = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().uiAutomation
        val now = android.os.SystemClock.uptimeMillis()
        for (action in listOf(android.view.MotionEvent.ACTION_DOWN, android.view.MotionEvent.ACTION_UP)) {
            val event = android.view.MotionEvent.obtain(now, now + action * 10L, action,
                point.x + location[0], point.y + location[1], 0)
            automation.injectInputEvent(event, true)
            event.recycle()
        }
        // Text selection can create unrelated popups; only the connection label must close.
        compose.onNodeWithText("Connected").assertDoesNotExist()
        compose.onNodeWithText("Keep me").assertExists()
    }

    @Test fun keyboardFocusEscapeAndFocusDepartureWithMultipleInstances() {
        lateinit var inputMode: androidx.compose.ui.input.InputModeManager
        compose.setContent { LelloTheme {
            inputMode = androidx.compose.ui.platform.LocalInputModeManager.current
            Row {
                LelloConnectionStatus(LelloConnectionState.Connected, labels = LelloConnectionLabels(title = "Server"))
                LelloConnectionStatus(LelloConnectionState.Disconnected, labels = LelloConnectionLabels(title = "Gateway"))
                LelloButton({}) { Text("Other") }
            }
        } }
        compose.runOnIdle { inputMode.requestInputMode(androidx.compose.ui.input.InputMode.Keyboard) }
        val server = compose.onNodeWithContentDescription("Server")
        server.performSemanticsAction(SemanticsActions.RequestFocus)
        compose.onNode(isPopup()).assertExists()
        server.performKeyInput { pressKey(Key.Escape) }
        compose.onNode(isPopup()).assertDoesNotExist()
        server.assertIsFocused()
        val gateway = compose.onNodeWithContentDescription("Gateway")
        gateway.performMouseInput { enter(center) }
        compose.runOnIdle { inputMode.requestInputMode(androidx.compose.ui.input.InputMode.Keyboard) }
        gateway.performSemanticsAction(SemanticsActions.RequestFocus)
        compose.onAllNodes(isPopup()).assertCountEquals(1)
        compose.onNodeWithText("Other").performSemanticsAction(SemanticsActions.RequestFocus)
        compose.onNode(isPopup()).assertDoesNotExist()
    }

    @Test fun hoverAndRtlNarrowWindowKeepTooltipVisible() {
        compose.setContent { LelloTheme {
            CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
                Row(Modifier.width(160.dp)) {
                    LelloConnectionStatus(LelloConnectionState.Connecting)
                }
            }
        } }
        val control = compose.onNodeWithContentDescription("Connection status")
        control.performMouseInput { enter(center) }
        compose.onNode(isPopup()).assertIsDisplayed()
        control.performMouseInput { exit() }
        compose.onNode(isPopup()).assertDoesNotExist()
    }
}
