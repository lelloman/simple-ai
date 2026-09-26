package com.lelloman.lellodesign

import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.test.*
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.junit4.StateRestorationTester
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test

class SharedControlsTest {
    @get:Rule val compose = createComposeRule()
    @Test fun stepProgressExposesCountAndClampsCompletion() {
        var count by mutableStateOf(4)
        compose.setContent { LelloTheme { LelloStepProgress(count, 9, androidx.compose.ui.Modifier.testTag("steps"), current=4) } }
        compose.onNodeWithTag("steps").assertRangeInfoEquals(androidx.compose.ui.semantics.ProgressBarRangeInfo(4f,0f..9f,8))
        compose.runOnIdle {count=12}
        compose.onNodeWithTag("steps").assertRangeInfoEquals(androidx.compose.ui.semantics.ProgressBarRangeInfo(9f,0f..9f,8))
    }
    @Test fun appearanceIsControlledAndCustomDoesNotClaimSystem() {
        var selected: LelloAppearance? = null
        compose.setContent { LelloTheme("green") {
            var state by remember { mutableStateOf<LelloAppearance?>(null) }
            LelloAppearanceSelector(state, { state=it; selected=it })
        } }
        compose.onNodeWithContentDescription("Appearance").assert(SemanticsMatcher.expectValue(androidx.compose.ui.semantics.SemanticsProperties.StateDescription,"Custom theme")).performClick()
        compose.onNodeWithText("Dark").performClick()
        compose.runOnIdle { assertEquals(LelloAppearance.Dark,selected) }
        compose.onNodeWithText("System").assertDoesNotExist()
        compose.onNodeWithContentDescription("Appearance").performClick()
        compose.onNodeWithText("System").performClick()
        compose.runOnIdle { assertEquals(LelloAppearance.System,selected) }
    }
    @Test fun themeDraftSurvivesRecreationAndSavesOnlyExplicitly() {
        val restore=StateRestorationTester(compose)
        var saved: String?=null
        restore.setContent { LelloTheme("green") {
            LelloThemeEditorDialog("Draft", false, listOf(LelloColorField("accent","Accent",0xff006c45.toInt())),
                {}, { name, _, colors -> saved=name; assertEquals(0xff006c45.toInt(),colors["accent"]) })
        } }
        compose.onNodeWithText("Name").performTextReplacement("My theme")
        restore.emulateSavedInstanceStateRestore()
        compose.onNodeWithText("My theme").assertIsDisplayed()
        compose.runOnIdle { assertNull(saved) }
        compose.onNodeWithText("Save").performClick()
        compose.runOnIdle { assertEquals("My theme",saved) }
    }
}
