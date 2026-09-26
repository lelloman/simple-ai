package com.lelloman.lellodesign

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.dp
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class OpenWorkspaceTest {
    @get:Rule val compose = createComposeRule()

    @Test fun wholeAccountIncludingPaddingActivatesOneTarget() {
        var opens = 0
        compose.setContent {
            LelloTheme {
                Box(Modifier.width(240.dp)) {
                    LelloAccount("Alex Morgan", { opens++ }, Modifier.testTag("account"), subtitle = "Personal workspace")
                }
            }
        }
        val account = compose.onNodeWithTag("account")
        account.assertWidthIsEqualTo(240.dp).assertHasClickAction()
        // Side/bottom padding is part of the target; the clipped round corner is not.
        account.performTouchInput { click(Offset(width - 3f, height / 2f)) }
        account.performTouchInput { click(Offset(width / 2f, height - 3f)) }
        compose.runOnIdle { assertEquals(2, opens) }
    }

    @Test fun materialBottomNavigationPreservesSelectionAcrossThemes() {
        var product by mutableStateOf("blue")
        var dark by mutableStateOf(false)
        var selected by mutableStateOf("home")
        compose.setContent {
            LelloTheme(product, dark) {
                LelloBottomNavigation(listOf(
                    LelloDestination("home", "Home") { Text("H") },
                    LelloDestination("settings", "Settings") { Text("S") },
                ), selected, { selected = it })
            }
        }
        for (family in listOf("blue", "green")) for (night in listOf(false, true)) {
            compose.runOnIdle { product = family; dark = night }
            compose.onNodeWithText("Home").performClick().assertIsSelected()
            compose.onNodeWithText("Settings").assertIsNotSelected().performClick().assertIsSelected()
            compose.runOnIdle { assertEquals("settings", selected) }
        }
    }

    @Test fun workspaceAdaptsGuttersWithoutReplacingContentState() {
        var wide by mutableStateOf(false)
        compose.setContent {
            LelloTheme {
                CompositionLocalProvider(LocalLelloWideLayout provides wide) {
                    LelloWorkspace {
                        var value by remember { mutableStateOf("") }
                        LelloTextField(value, { value = it }, { Text("Draft") }, Modifier.testTag("draft"))
                    }
                }
            }
        }
        compose.onNodeWithTag("draft").assertLeftPositionInRootIsEqualTo(16.dp).performTextInput("Keep this draft")
        compose.runOnIdle { wide = true }
        compose.onNodeWithTag("draft").assertLeftPositionInRootIsEqualTo(32.dp)
        compose.onNodeWithText("Keep this draft").assertExists()
    }
}
