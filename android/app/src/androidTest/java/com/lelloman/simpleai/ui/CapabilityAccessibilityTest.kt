package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.ui.theme.SimpleAITheme
import org.junit.Rule
import org.junit.Test

class CapabilityAccessibilityTest {
    @get:Rule val compose = createComposeRule()
    @Test fun narrowLargeTextCardKeepsContextualActionsReachable() {
        compose.setContent {
            CompositionLocalProvider(LocalDensity provides Density(LocalDensity.current.density, 2f)) {
                SimpleAITheme {
                    Box(Modifier.width(280.dp)) {
                        CapabilityCard("Translation", "🌐", "Translate text", CapabilityStatus.Ready,
                            onDelete = {}, onTest = {})
                    }
                }
            }
        }
        compose.onNodeWithText("Delete Translation").assertIsDisplayed().assertHasClickAction()
        compose.onNodeWithText("Test Translation").assertIsDisplayed().assertHasClickAction()
        compose.onNodeWithText("Ready").assertExists()
        compose.onNodeWithText("🌐").assertDoesNotExist()
    }
}
