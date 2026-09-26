package com.lelloman.lellodesign

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import org.junit.Rule
import org.junit.Test

class ScaffoldTest {
    @get:Rule val compose = createComposeRule()

    @Test fun drawerNavigatesAndClosesAndAccountStaysOutOfHeader() {
        compose.setContent {
            var page by remember { mutableStateOf("home") }
            LelloTheme {
                Box(Modifier.width(400.dp)) {
                    LelloScaffold("Sample", page, destinations(), page, { page = it },
                        account = { LelloAccount("Test User", {}) }) { padding -> Text("Content: $page", Modifier.padding(padding)) }
                }
            }
        }
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
        compose.onNodeWithContentDescription("Open navigation").performClick()
        compose.onNodeWithContentDescription("Account: Test User").assertIsDisplayed()
        androidx.test.espresso.Espresso.pressBack()
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
        compose.onNodeWithText("Content: home").assertIsDisplayed()
        compose.onNodeWithContentDescription("Open navigation").performClick()
        compose.onNodeWithContentDescription("Settings").performClick()
        compose.onNodeWithText("Content: settings").assertIsDisplayed()
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
    }

    @Test fun navigationModeSwitchKeepsFieldStateAndMaterialThemeUpdates() {
        var bottom by mutableStateOf(false)
        var dark by mutableStateOf(false)
        compose.setContent {
            LelloTheme("purplegreen", dark) {
                Box(Modifier.width(400.dp)) {
                    LelloScaffold("Sample", "Home", destinations(), "home", {},
                        mobileNavigation = if (bottom) LelloMobileNavigation.Bottom else LelloMobileNavigation.Drawer) { padding ->
                        var value by remember { mutableStateOf("") }
                        LelloTextField(value, { value = it }, { Text("Name") }, Modifier.padding(padding))
                        check(MaterialTheme.colorScheme.primary == LelloPalettes.forProduct("purplegreen", dark)["primary"])
                    }
                }
            }
        }
        compose.onNodeWithText("Name").performTextInput("Persistent draft")
        compose.runOnIdle { bottom = true; dark = true }
        compose.onNodeWithText("Persistent draft").assertIsDisplayed()
        compose.onNodeWithContentDescription("Open navigation").assertDoesNotExist()
        compose.onNodeWithText("Settings").assertIsDisplayed()
    }

    @Test fun resizingAndCollapsingKeepContentAndAccountState() {
        var wide by mutableStateOf(false)
        var accountCompositions = 0
        compose.setContent {
            LelloTheme {
                Box(Modifier.wrapContentSize(androidx.compose.ui.Alignment.TopStart, unbounded = true).requiredWidth(if (wide) 900.dp else 400.dp).requiredHeight(600.dp)) {
                    LelloScaffold("Sample", "Home", destinations(), "home", {},
                        account = { compact ->
                            remember { accountCompositions++ }
                            LelloAccount("Test User", {}, compact = compact)
                        }) { padding ->
                        var value by remember { mutableStateOf("") }
                        LelloTextField(value, { value = it }, { Text("Draft") }, Modifier.padding(padding))
                    }
                }
            }
        }
        compose.onNodeWithText("Draft").performTextInput("Keep me")
        compose.runOnIdle { wide = true }
        compose.onNodeWithContentDescription("Collapse sidebar").performClick()
        compose.onNodeWithContentDescription("Expand sidebar").assertExists()
        compose.onNodeWithText("Keep me").assertExists()
        compose.runOnIdle { wide = false }
        compose.onNodeWithContentDescription("Open navigation").assertExists()
        compose.onNodeWithText("Keep me").assertExists()
        compose.runOnIdle { org.junit.Assert.assertEquals(1, accountCompositions) }
    }

    @Test fun combinedNavigationUsesIndependentTabsAndDrawerBackDoesNotNavigate() {
        var backCalls = 0
        var detail by mutableStateOf(false)
        compose.setContent {
            var page by remember { mutableStateOf("home") }
            androidx.activity.compose.BackHandler(detail) { backCalls++; detail = false }
            LelloTheme {
                Box(Modifier.width(400.dp)) {
                    LelloScaffold("Sample", "Page", destinations(), page, { page = it },
                        mobileNavigation = LelloMobileNavigation.DrawerAndBottom,
                        bottomDestinations = listOf(destinations().first()),
                        onBack = if (detail) ({ backCalls++; detail = false }) else null,
                        labels = LelloScaffoldLabels(navigateBack = "Go back"),
                        account = { LelloAccount("Test User", {}) }) { padding ->
                        Text("Content: $page", Modifier.padding(padding))
                    }
                }
            }
        }
        compose.onNode(hasText("Home") and !hasContentDescription("Home")).assertIsDisplayed()
        compose.onNodeWithText("Settings").assertIsNotDisplayed()
        compose.onNodeWithContentDescription("Open navigation").performClick()
        compose.onNodeWithContentDescription("Settings").performClick()
        compose.onNodeWithText("Content: settings").assertIsDisplayed()
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
        compose.onNode(hasText("Home") and !hasContentDescription("Home")).performClick()
        compose.onNodeWithText("Content: home").assertIsDisplayed()
        compose.onNodeWithContentDescription("Open navigation").performClick()
        // Moving to a detail route while the drawer is open must not steal system Back.
        compose.runOnIdle { detail = true }
        androidx.test.espresso.Espresso.pressBack()
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
        compose.runOnIdle { org.junit.Assert.assertEquals(0, backCalls) }
        compose.onNodeWithContentDescription("Open navigation").assertDoesNotExist()
        compose.onNodeWithContentDescription("Go back").performClick()
        compose.runOnIdle { org.junit.Assert.assertEquals(1, backCalls) }
        compose.onNodeWithContentDescription("Open navigation").assertIsDisplayed()
    }

    @Test fun combinedNavigationResizeAndTabChangesPreserveDraft() {
        var wide by mutableStateOf(false)
        var tabs by mutableStateOf(listOf(destinations().first()))
        compose.setContent {
            LelloTheme {
                Box(Modifier.wrapContentSize(androidx.compose.ui.Alignment.TopStart, unbounded = true)
                    .requiredWidth(if (wide) 900.dp else 400.dp).requiredHeight(600.dp)) {
                    LelloScaffold("Sample", "Detail", destinations(), "home", {},
                        mobileNavigation = LelloMobileNavigation.DrawerAndBottom,
                        bottomDestinations = tabs, onBack = {}) { padding ->
                        var draft by remember { mutableStateOf("") }
                        LelloTextField(draft, { draft = it }, { Text("Draft") }, Modifier.padding(padding))
                    }
                }
            }
        }
        compose.onNodeWithText("Draft").performTextInput("Keep draft")
        compose.runOnIdle { tabs = emptyList() }
        compose.onNodeWithText("Home").assertIsNotDisplayed()
        compose.runOnIdle { wide = true }
        compose.onNodeWithContentDescription("Collapse sidebar").assertExists()
        compose.onNodeWithContentDescription("Back").assertExists()
        compose.onNodeWithText("Settings").assertExists()
        compose.runOnIdle { wide = false; tabs = listOf(destinations().last()) }
        compose.onNode(hasText("Settings") and !hasContentDescription("Settings")).assertIsDisplayed()
        compose.onNodeWithText("Keep draft").assertExists()
        compose.onNodeWithContentDescription("Back").assertIsDisplayed()
    }

    @Test fun suppliedDrawerStateSupportsHostNavigationAndNativeDismissal() {
        lateinit var drawer: DrawerState
        lateinit var scope: CoroutineScope
        var page by mutableStateOf("home")
        var wide by mutableStateOf(false)
        compose.setContent {
            drawer = rememberDrawerState(DrawerValue.Closed)
            scope = rememberCoroutineScope()
            LelloTheme {
                Box(Modifier.wrapContentSize(androidx.compose.ui.Alignment.TopStart, unbounded = true)
                    .requiredWidth(if (wide) 900.dp else 400.dp).requiredHeight(600.dp)) {
                    LelloScaffold("Sample", "Page", destinations(), page, { page = it },
                        mobileNavigation = LelloMobileNavigation.DrawerAndBottom,
                        bottomDestinations = listOf(destinations().first()), drawerState = drawer,
                        account = { LelloAccount("Test User", {}) }) { padding ->
                        Column(Modifier.padding(padding)) {
                            Text("Content: $page")
                            var draft by remember { mutableStateOf("") }
                            LelloTextField(draft, { draft = it }, { Text("Draft") })
                        }
                    }
                }
            }
        }
        compose.onNodeWithText("Draft").performTextInput("Keep draft")
        compose.onNodeWithContentDescription("Open navigation").performClick()
        compose.runOnIdle { org.junit.Assert.assertTrue(drawer.isOpen) }
        // A host such as Casina's diagnostics bridge can close before changing routes.
        compose.runOnIdle { scope.launch { drawer.close(); page = "settings" } }
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
        compose.onNodeWithText("Content: settings").assertIsDisplayed()
        compose.onNodeWithText("Keep draft").assertExists()
        compose.runOnIdle { scope.launch { drawer.open() } }
        compose.waitForIdle()
        compose.onNodeWithContentDescription("Account: Test User").assertIsDisplayed()
        androidx.test.espresso.Espresso.pressBack()
        compose.runOnIdle { org.junit.Assert.assertTrue(drawer.isClosed) }
        compose.onNodeWithText("Content: settings").assertIsDisplayed()
        compose.onNodeWithContentDescription("Open navigation").performClick()
        compose.onNodeWithContentDescription("Home").performClick()
        compose.runOnIdle { org.junit.Assert.assertTrue(drawer.isClosed) }
        compose.onNodeWithText("Content: home").assertIsDisplayed()
        compose.runOnIdle { scope.launch { drawer.open() } }
        compose.waitForIdle()
        compose.runOnIdle { wide = true }
        compose.waitForIdle()
        compose.runOnIdle { org.junit.Assert.assertTrue(drawer.isClosed); wide = false }
        compose.onNodeWithContentDescription("Account: Test User").assertIsNotDisplayed()
        compose.onNodeWithText("Keep draft").assertExists()
    }

    private fun destinations() = listOf(
        LelloDestination("home", "Home") { Text("H") },
        LelloDestination("settings", "Settings") { Text("S") },
    )
}
