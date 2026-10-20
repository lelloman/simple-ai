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
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.junit4.createAndroidComposeRule
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
@Config(sdk = [35], qualifiers = "w411dp-h891dp-xxhdpi")
class ConnectedAppsScreenshotTest {
    @get:Rule val compose = createAndroidComposeRule<ComponentActivity>()

    private data class App(
        val packageName: String,
        val label: String?,
        val letter: String?,
        val color: Int,
        val approved: Boolean,
    )

    private val apps = mutableStateOf(emptyList<App>())
    private val dark = mutableStateOf(false)
    private val fontScale = mutableStateOf(1f)
    private val icons = mutableMapOf<String, ImageBitmap>()

    @Test fun captureScreenshotMatrix() {
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale.value)) {
                SimpleAITheme(darkTheme = dark.value) {
                    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                        ConnectedAppsContent(
                            clients = apps.value.map { ClientAccess.Client(it.packageName, it.packageName, it.approved) },
                            onSetApproved = { _, _ -> },
                            metadata = { packages -> metadataFor(packages) },
                        )
                    }
                }
            }
        }

        val empty = emptyList<App>()
        val pending = listOf(
            app("com.example.maps", "Maps", "M", 0xFF4285F4.toInt()),
            app("com.example.photos", "Photos", "P", 0xFFEA4335.toInt()),
        )
        val approved = listOf(
            app("com.example.chat", "Chat", "C", 0xFF34A853.toInt(), approved = true),
            app("com.example.calendar", "Calendar", "31", 0xFFFBBC05.toInt(), approved = true),
        )
        val mixed = listOf(
            app("com.example.drive", "Drive", "D", 0xFF4285F4.toInt()),
            app("com.example.chat", "Chat", "C", 0xFF34A853.toInt(), approved = true),
            app("com.example.photos", "Photos", "P", 0xFFEA4335.toInt()),
            app("com.example.calendar", "Calendar", "31", 0xFFFBBC05.toInt(), approved = true),
        )

        for (isDark in listOf(false, true)) {
            val name = if (isDark) "dark" else "light"
            capture("after-$name-empty", empty, isDark)
            capture("after-$name-pending", pending, isDark)
            capture("after-$name-approved", approved, isDark)
            capture("after-$name-mixed", mixed, isDark)
        }
    }

    @Test fun captureManyEntries() {
        val many = listOf(
            "Maps" to "map", "Chat" to "chat", "Photos" to "photo", "Drive" to "drive",
            "Camera" to "camera", "Music" to "music", "Notes" to "note", "Calendar" to "cal",
            "Clock" to "clock", "Files" to "file", "Mail" to "mail", "Wallet" to "wallet",
        ).mapIndexed { index, (label, key) ->
            app("com.example.$key", label, label.first().toString(), palette[index % palette.size], approved = index % 3 == 2)
        }
        setContent()
        capture("after-light-many", many, isDark = false)
        capture("after-dark-many", many, isDark = true)
    }

    private fun setContent() {
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale.value)) {
                SimpleAITheme(darkTheme = dark.value) {
                    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                        ConnectedAppsContent(
                            clients = apps.value.map { ClientAccess.Client(it.packageName, it.packageName, it.approved) },
                            onSetApproved = { _, _ -> },
                            metadata = { packages -> metadataFor(packages) },
                        )
                    }
                }
            }
        }
    }

    private fun capture(name: String, value: List<App>, isDark: Boolean, scale: Float = 1f) {
        apps.value = value
        dark.value = isDark
        fontScale.value = scale
        compose.waitForIdle()
        val view = compose.activity.window.decorView
        val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
        view.draw(Canvas(bitmap))
        val out = File("build/screenshots/ai-52/$name.png")
        out.parentFile?.mkdirs()
        out.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    private fun metadataFor(packages: List<String>): AppMetadata {
        val packageName = packages.firstOrNull()
        val app = apps.value.firstOrNull { it.packageName == packageName }
        return if (app?.label == null) {
            AppMetadata(null, null)
        } else {
            AppMetadata(app.label, icons.getOrPut(app.packageName) { icon(app) })
        }
    }

    private fun app(
        packageName: String,
        label: String,
        letter: String?,
        color: Int,
        approved: Boolean = false,
    ) = App(packageName, label, letter, color, approved)

    private fun icon(app: App): ImageBitmap {
        val size = 144
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = app.color }
        canvas.drawRoundRect(0f, 0f, size.toFloat(), size.toFloat(), 32f, 32f, paint)
        paint.color = Color.WHITE
        paint.textSize = if (app.letter!!.length == 1) size * 0.55f else size * 0.42f
        paint.typeface = Typeface.DEFAULT_BOLD
        paint.textAlign = Paint.Align.CENTER
        canvas.drawText(app.letter, size / 2f, size * 0.68f, paint)
        return bitmap.asImageBitmap()
    }

    private companion object {
        val palette = listOf(
            0xFF4285F4.toInt(),
            0xFFEA4335.toInt(),
            0xFF34A853.toInt(),
            0xFFFBBC05.toInt(),
            0xFF8E24AA.toInt(),
            0xFF00897B.toInt(),
        )
    }
}
