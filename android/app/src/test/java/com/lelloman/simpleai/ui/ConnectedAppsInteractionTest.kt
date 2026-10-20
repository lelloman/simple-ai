package com.lelloman.simpleai.ui

import androidx.activity.ComponentActivity
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.getUnclippedBoundsInRoot
import androidx.compose.ui.test.hasScrollAction
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToIndex
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.lelloman.simpleai.R
import com.lelloman.simpleai.access.ClientAccess
import com.lelloman.simpleai.ui.theme.SimpleAITheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h891dp-xxhdpi")
class ConnectedAppsInteractionTest {
    @get:Rule val compose = createAndroidComposeRule<ComponentActivity>()

    private fun setContent(
        clients: () -> List<ClientAccess.Client>,
        onSetApproved: (ClientAccess.Client, Boolean) -> Unit = { _, _ -> },
        labelFor: (String) -> String? = { it.substringAfterLast('.').replaceFirstChar(Char::uppercase) },
        fontScale: Float = 1f,
    ) {
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale)) {
                SimpleAITheme(darkTheme = false) {
                    ConnectedAppsContent(
                        clients = clients(),
                        onSetApproved = onSetApproved,
                        metadata = { packages ->
                            AppMetadata(packages.firstOrNull()?.let(labelFor), null)
                        },
                    )
                }
            }
        }
    }

    @Test fun emptyStateExplainsHowAppsAppear() {
        setContent(clients = { emptyList() })
        compose.onNodeWithText(compose.activity.getString(R.string.apps_empty_title)).assertIsDisplayed()
        compose.onNodeWithText(compose.activity.getString(R.string.apps_empty_hint)).assertIsDisplayed()
    }

    @Test fun pendingAppsAppearBeforeApprovedApps() {
        setContent(clients = {
            listOf(
                ClientAccess.Client("approved", "com.example.calendar", true),
                ClientAccess.Client("pending", "com.example.maps", false),
            )
        })
        val pendingTop = compose.onNodeWithText("Maps").getUnclippedBoundsInRoot().top
        val approvedTop = compose.onNodeWithText("Calendar").getUnclippedBoundsInRoot().top
        assertTrue("pending app should be above approved app", pendingTop < approvedTop)
    }

    @Test fun approvingUpdatesTagAndActionImmediately() {
        val clients = mutableStateOf(listOf(ClientAccess.Client("maps", "com.example.maps", false)))
        var action: Pair<String, Boolean>? = null
        setContent(
            clients = { clients.value },
            onSetApproved = { client, approved ->
                action = client.identity to approved
                clients.value = clients.value.map {
                    if (it.identity == client.identity) it.copy(approved = approved) else it
                }
            },
        )
        compose.onNodeWithText("Approve").assertIsDisplayed().performClick()
        compose.onNodeWithText(compose.activity.getString(R.string.ui_approved)).assertIsDisplayed()
        compose.onNodeWithText(compose.activity.getString(R.string.apps_revoke)).assertIsDisplayed()
        compose.onNodeWithText(compose.activity.getString(R.string.apps_approve)).assertDoesNotExist()
        assertEquals("maps" to true, action)
    }

    @Test fun revokingUpdatesTagAndActionImmediately() {
        val clients = mutableStateOf(listOf(ClientAccess.Client("maps", "com.example.maps", true)))
        var action: Pair<String, Boolean>? = null
        setContent(
            clients = { clients.value },
            onSetApproved = { client, approved ->
                action = client.identity to approved
                clients.value = clients.value.map {
                    if (it.identity == client.identity) it.copy(approved = approved) else it
                }
            },
        )
        compose.onNodeWithText(compose.activity.getString(R.string.apps_revoke)).performClick()
        compose.onNodeWithText(compose.activity.getString(R.string.apps_waiting)).assertIsDisplayed()
        compose.onNodeWithText(compose.activity.getString(R.string.apps_approve)).assertIsDisplayed()
        compose.onNodeWithText(compose.activity.getString(R.string.apps_revoke)).assertDoesNotExist()
        assertEquals("maps" to false, action)
    }

    @Test fun missingMetadataUsesFallbackNameAndPackageIdentifier() {
        setContent(clients = { listOf(ClientAccess.Client("missing", "com.example.uninstalled", false)) }, labelFor = { null })
        compose.onNodeWithText(compose.activity.getString(R.string.apps_unknown_name)).assertIsDisplayed()
        compose.onNodeWithText("com.example.uninstalled").assertIsDisplayed()
        compose.onNodeWithContentDescription("Approve access for com.example.uninstalled").assertIsDisplayed()
    }

    @Test fun actionsHaveTargetSpecificAccessibilityLabels() {
        setContent(clients = { listOf(ClientAccess.Client("maps", "com.example.maps", false)) }, labelFor = { "Maps" })
        compose.onNodeWithContentDescription("Approve access for Maps").assertIsDisplayed()
    }

    @Test fun longLabelsAtLargeFontScaleRemainVisibleAndSeparated() {
        val longPackage = "com.lelloman.familybudgetplanner.expensetracker, com.lelloman.familybudgetplanner.widget"
        setContent(
            clients = { listOf(ClientAccess.Client("long", longPackage, false)) },
            labelFor = { "Family Budget Planner & Expense Tracker" },
            fontScale = 2f,
        )
        val name = compose.onNodeWithText("Family Budget Planner & Expense Tracker")
        val tag = compose.onNodeWithText(compose.activity.getString(R.string.apps_waiting))
        val button = compose.onNodeWithText(compose.activity.getString(R.string.apps_approve))
        name.assertIsDisplayed()
        tag.assertIsDisplayed()
        button.assertIsDisplayed()
        assertTrue(button.getUnclippedBoundsInRoot().top >= tag.getUnclippedBoundsInRoot().bottom)
        val buttonBounds = button.getUnclippedBoundsInRoot()
        assertTrue(buttonBounds.bottom - buttonBounds.top >= 48.dp)
    }

    @Test fun manyEntriesCanBeScrolledToTheLastApprovedApp() {
        val clients = (1..24).map { index ->
            ClientAccess.Client("client-$index", "com.example.app$index", approved = index % 2 == 0)
        }
        setContent(clients = { clients })
        compose.onNode(hasScrollAction()).performScrollToIndex(clients.size)
        compose.onNodeWithText("App24").assertIsDisplayed()
    }
}
