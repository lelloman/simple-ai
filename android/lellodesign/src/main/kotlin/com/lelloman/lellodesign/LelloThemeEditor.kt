package com.lelloman.lellodesign

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import kotlin.math.roundToInt

data class LelloColorField(val id: String, val label: String, val argb: Int)
data class LelloThemeEditorLabels(val title: String = "Edit theme", val name: String = "Name",
    val dark: String = "Dark palette", val darkDescription: String = "Use light system icons on dark surfaces.",
    val save: String = "Save", val cancel: String = "Cancel", val delete: String = "Delete",
    val apply: String = "Apply", val red: String = "Red", val green: String = "Green", val blue: String = "Blue")

/** Product supplies the color roles. Dialog drafts survive recreation; save is explicit. */
@Composable
fun LelloThemeEditorDialog(initialName: String, initialDark: Boolean, fields: List<LelloColorField>,
    onDismiss: () -> Unit, onSave: (String, Boolean, Map<String, Int>) -> Unit,
    onDelete: (() -> Unit)? = null, labels: LelloThemeEditorLabels = LelloThemeEditorLabels()) {
    var name by rememberSaveable { mutableStateOf(initialName) }
    var dark by rememberSaveable { mutableStateOf(initialDark) }
    var colors by rememberSaveable { mutableStateOf(fields.map { it.argb }.toIntArray()) }
    var selected by rememberSaveable { mutableStateOf<Int?>(null) }
    Dialog(onDismissRequest = onDismiss) {
        Surface(Modifier.fillMaxWidth().heightIn(max = 720.dp), shape = MaterialTheme.shapes.large) {
            Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(labels.title, style = MaterialTheme.typography.headlineSmall)
                Column(Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    LelloTextField(name, { name = it }, { Text(labels.name) }, Modifier.fillMaxWidth())
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) { Text(labels.dark); Text(labels.darkDescription, style = MaterialTheme.typography.bodySmall) }
                        Switch(dark, { dark = it }, Modifier.semantics { contentDescription = labels.dark })
                    }
                    fields.forEachIndexed { index, field ->
                        LelloColorSetting(field.label, colors[index], { selected = index })
                    }
                }
                @OptIn(ExperimentalLayoutApi::class)
                FlowRow(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    if (onDelete != null) LelloTextButton(onDelete) { Text(labels.delete, color = MaterialTheme.colorScheme.error) }
                    LelloTextButton(onDismiss) { Text(labels.cancel) }
                    LelloTextButton({ onSave(name.trim(), dark, fields.mapIndexed { i, f -> f.id to colors[i] }.toMap()) }, enabled = name.isNotBlank()) { Text(labels.save) }
                }
            }
        }
    }
    selected?.let { index ->
        LelloColorPickerDialog(fields[index].label, colors[index], { selected = null },
            { color -> colors = colors.copyOf().also { it[index] = color }; selected = null }, labels)
    }
}

@Composable
fun LelloColorSetting(label: String, argb: Int, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).clickable(onClick = onClick)
        .heightIn(min = 48.dp).padding(8.dp).semantics(mergeDescendants = true) {},
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Box(Modifier.size(32.dp).background(Color(argb), MaterialTheme.shapes.small)
            .border(1.dp, MaterialTheme.colorScheme.outline, MaterialTheme.shapes.small))
        Column(Modifier.weight(1f)) { Text(label); Text(hex(argb), style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
fun LelloColorPickerDialog(title: String, initialColor: Int, onDismiss: () -> Unit, onConfirm: (Int) -> Unit,
    labels: LelloThemeEditorLabels = LelloThemeEditorLabels()) {
    var r by rememberSaveable { mutableFloatStateOf((initialColor ushr 16 and 255).toFloat()) }
    var g by rememberSaveable { mutableFloatStateOf((initialColor ushr 8 and 255).toFloat()) }
    var b by rememberSaveable { mutableFloatStateOf((initialColor and 255).toFloat()) }
    val color = (initialColor and 0xff000000.toInt()) or (r.roundToInt() shl 16) or (g.roundToInt() shl 8) or b.roundToInt()
    AlertDialog(onDismissRequest = onDismiss, shape = MaterialTheme.shapes.large, title = { Text(title) }, text = {
        Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(Modifier.fillMaxWidth().height(64.dp).background(Color(color), MaterialTheme.shapes.medium)
                .border(1.dp, MaterialTheme.colorScheme.outline, MaterialTheme.shapes.medium))
            listOf(Triple(labels.red,r,{ v: Float -> r=v }),Triple(labels.green,g,{ v: Float -> g=v }),Triple(labels.blue,b,{ v: Float -> b=v })).forEach { (label,value,change) ->
                Text("$label: ${value.roundToInt()}", style = MaterialTheme.typography.labelMedium)
                Slider(value, change, valueRange = 0f..255f, steps = 254, modifier = Modifier.semantics { contentDescription = label })
            }
            Text(hex(color))
        }
    }, confirmButton = { LelloTextButton({ onConfirm(color) }) { Text(labels.apply) } },
        dismissButton = { LelloTextButton(onDismiss) { Text(labels.cancel) } })
}
private fun hex(argb: Int) = "#" + (argb and 0xffffff).toString(16).uppercase().padStart(6,'0')
