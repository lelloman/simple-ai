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
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class CapabilityAccessibilityTest {
    @get:Rule val compose = createComposeRule()
    @Test fun narrowLargeTextModelRowsKeepNavigationReachable() {
        var selected: String? = null
        compose.setContent {
            CompositionLocalProvider(LocalDensity provides Density(LocalDensity.current.density, 2f)) {
                SimpleAITheme {
                    Box(Modifier.width(280.dp)) {
                        ModelsContent(CapabilitiesState(voiceCommandsStatus = CapabilityStatus.Ready),
                            onOpenModel = { selected = it }, onLanguages = {})
                    }
                }
            }
        }
        compose.onNodeWithText("Voice Commands").assertIsDisplayed().performClick()
        compose.runOnIdle { assertEquals("voice", selected) }
        compose.onNodeWithText("Languages").performScrollTo().assertIsDisplayed().assertHasClickAction()
        compose.onNodeWithText("Cloud AI").assertDoesNotExist()
        compose.onNodeWithText("Service connected").assertDoesNotExist()
    }
    @Test fun firstRunModelsOpensLanguageManagementWithoutDownloads() {
        var opened = false
        compose.setContent {
            SimpleAITheme { ModelsContent(CapabilitiesState(), onOpenModel = {}, onLanguages = { opened = true }) }
        }
        compose.onNodeWithText("Languages").performClick()
        compose.runOnIdle { assertEquals(true, opened) }
    }
}
