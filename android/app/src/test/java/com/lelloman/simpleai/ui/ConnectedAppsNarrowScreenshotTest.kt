package com.lelloman.simpleai.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import androidx.activity.ComponentActivity
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.hasScrollAction
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.performScrollToIndex
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import com.lelloman.simpleai.access.ClientAccess
import com.lelloman.simpleai.ui.theme.SimpleAITheme
import java.io.File
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w320dp-h640dp-xxhdpi")
class ConnectedAppsNarrowScreenshotTest {
    @get:Rule val compose = createAndroidComposeRule<ComponentActivity>()

    private val dark = mutableStateOf(false)

    @Test fun captureLongLabelsAtLargeFontScale() {
        val clients = listOf(
            ClientAccess.Client(
                "long-1",
                "com.lelloman.familybudgetplanner.expensetracker, com.lelloman.familybudgetplanner.widget",
                false,
            ),
            ClientAccess.Client("long-2", "com.example.collaborative.research.assistant.production", true),
        )
        val metadata = mapOf(
            "com.lelloman.familybudgetplanner.expensetracker" to AppMetadata(
                "Family Budget Planner & Expense Tracker",
                icon("F", 0xFF8E24AA.toInt()),
            ),
            "com.example.collaborative.research.assistant.production" to AppMetadata(
                "Collaborative Research Assistant",
                icon("R", 0xFF00897B.toInt()),
            ),
        )
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale = 2f)) {
                SimpleAITheme(darkTheme = dark.value) {
                    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                        ConnectedAppsContent(
                            clients = clients,
                            onSetApproved = { _, _ -> },
                            metadata = { packages -> metadata[packages.firstOrNull()] ?: AppMetadata(null, null) },
                        )
                    }
                }
            }
        }
        for (isDark in listOf(false, true)) {
            dark.value = isDark
            compose.waitForIdle()
            val prefix = if (isDark) "dark" else "light"
            compose.onNode(hasScrollAction()).performScrollToIndex(1)
            capture("after-$prefix-long-large-font")
            compose.onNode(hasScrollAction()).performScrollToIndex(2)
            capture("after-$prefix-long-large-font-actions")
        }
    }

    private fun capture(name: String) {
        compose.waitForIdle()
        val view = compose.activity.window.decorView
        val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
        view.draw(Canvas(bitmap))
        val out = File("build/screenshots/ai-52/$name.png")
        out.parentFile?.mkdirs()
        out.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    private fun icon(letter: String, color: Int): ImageBitmap {
        val size = 144
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color }
        canvas.drawRoundRect(0f, 0f, size.toFloat(), size.toFloat(), 32f, 32f, paint)
        paint.color = Color.WHITE
        paint.textSize = size * 0.55f
        paint.typeface = Typeface.DEFAULT_BOLD
        paint.textAlign = Paint.Align.CENTER
        canvas.drawText(letter, size / 2f, size * 0.7f, paint)
        return bitmap.asImageBitmap()
    }
}
