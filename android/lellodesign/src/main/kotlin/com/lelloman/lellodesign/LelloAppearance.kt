package com.lelloman.lellodesign

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.dp

enum class LelloAppearance { Light, Dark, System }
data class LelloAppearanceLabels(val title: String = "Appearance", val light: String = "Light",
    val dark: String = "Dark", val system: String = "System") {
    fun label(value: LelloAppearance): String = when (value) {
        LelloAppearance.Light -> light; LelloAppearance.Dark -> dark; LelloAppearance.System -> system
    }
}

/** Controlled selector. The host owns storage and system appearance resolution.
 * A null selection represents a product-specific custom theme, never a fake System state.
 */
@Composable
fun LelloAppearanceSelector(selected: LelloAppearance?, onSelected: (LelloAppearance) -> Unit,
    modifier: Modifier = Modifier, labels: LelloAppearanceLabels = LelloAppearanceLabels(),
    customLabel: String = "Custom theme") {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier) {
        IconButton(onClick = { expanded = true }, modifier = Modifier.semantics {
            contentDescription = labels.title
            stateDescription = selected?.let(labels::label) ?: customLabel
        }) { AppearanceGlyph(selected) }
        DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
            LelloAppearance.entries.forEach { appearance ->
                DropdownMenuItem(text = { Text(labels.label(appearance)) }, leadingIcon = { AppearanceGlyph(appearance) },
                    trailingIcon = { if (selected == appearance) Text("✓", Modifier.clearAndSetSemantics {}) },
                    modifier = Modifier.semantics { this.selected = selected == appearance },
                    onClick = { expanded = false; onSelected(appearance) })
            }
        }
    }
}

@Composable
private fun AppearanceGlyph(value: LelloAppearance?) {
    val color = LocalContentColor.current
    Canvas(Modifier.size(20.dp)) {
        val w = size.width; val h = size.height; val stroke = 1.6.dp.toPx()
        when (value) {
            LelloAppearance.Light -> {
                drawCircle(color, w*.2f, style = Stroke(stroke))
                for (i in 0..7) {
                    val angle = i * kotlin.math.PI / 4
                    val v = Offset(kotlin.math.cos(angle).toFloat(), kotlin.math.sin(angle).toFloat())
                    drawLine(color, center + v*(w*.35f), center + v*(w*.46f), stroke)
                }
            }
            LelloAppearance.Dark -> {
                val p = androidx.compose.ui.graphics.Path().apply {
                    moveTo(w*.75f,h*.7f); cubicTo(w*.25f,h*.9f,w*.1f,h*.35f,w*.45f,h*.15f)
                    cubicTo(w*.3f,h*.5f,w*.55f,h*.75f,w*.75f,h*.7f); close()
                }
                drawPath(p,color, style = Stroke(stroke))
            }
            LelloAppearance.System -> {
                drawRect(color, Offset(w*.1f,h*.15f), androidx.compose.ui.geometry.Size(w*.8f,h*.55f), style = Stroke(stroke))
                drawLine(color, Offset(w*.5f,h*.7f), Offset(w*.5f,h*.9f),stroke)
                drawLine(color, Offset(w*.3f,h*.9f), Offset(w*.7f,h*.9f),stroke)
            }
            null -> { drawCircle(color,w*.36f,style=Stroke(stroke)); drawCircle(color,w*.12f) }
        }
    }
}
