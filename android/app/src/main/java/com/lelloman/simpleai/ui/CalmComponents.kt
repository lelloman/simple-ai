package com.lelloman.simpleai.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.KeyboardArrowRight
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.lelloman.lellodesign.LelloDimensions
import com.lelloman.lellodesign.LelloElevation
import com.lelloman.lellodesign.LelloTextButton
import com.lelloman.lellodesign.LelloTextStyles
import com.lelloman.lellodesign.LocalLelloPalette
import com.lelloman.lellodesign.familyRule
import com.lelloman.lellodesign.lelloShadow

/** Status tones for [StatusPill]; each maps to the palette's `<tone>-container` roles. */
internal enum class PillTone { Success, Warning, Error, Info, Neutral }

/**
 * Soft status pill (LelloDesign 0.5 interim pattern until Compose ships a badge): a fully rounded
 * container fill, 12 sp medium text and a 6 dp dot in the tone's border color, with no outline.
 */
@Composable
internal fun StatusPill(text: String, tone: PillTone, modifier: Modifier = Modifier) {
    val palette = LocalLelloPalette.current
    val (container, content, dot) = when (tone) {
        PillTone.Neutral -> Triple(palette["neutral"], palette["on-neutral"], palette["border-control"])
        else -> {
            val key = tone.name.lowercase()
            Triple(palette["$key-container"], palette["on-$key-container"], palette["$key-border"])
        }
    }
    Surface(modifier, shape = CircleShape, color = container, contentColor = content) {
        Row(
            Modifier.heightIn(min = 20.dp).padding(horizontal = LelloDimensions.space2, vertical = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Box(Modifier.size(6.dp).background(dot, CircleShape))
            Text(text, style = LelloTextStyles.overline.copy(lineHeight = 16.sp), maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

/** Quiet row divider used between list rows inside open sections. */
@Composable
internal fun RowDivider() = HorizontalDivider(thickness = 1.dp, color = LocalLelloPalette.current["border-subtle"])

/**
 * A navigable row: body title, optional secondary caption, optional trailing status and a chevron.
 * The whole row is the touch target.
 */
@Composable
internal fun NavigationRow(
    title: String,
    supporting: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    status: (@Composable () -> Unit)? = null,
    @DrawableRes trailingIcon: Int? = null,
) {
    Row(
        modifier.fillMaxWidth().heightIn(min = 64.dp).clickable(onClick = onClick).padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space3),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
            if (supporting != null) Text(supporting, style = MaterialTheme.typography.bodyMedium,
                color = LocalLelloPalette.current["text-secondary"])
            if (status != null) Box(Modifier.padding(top = 4.dp)) { status() }
        }
        if (trailingIcon != null) {
            Icon(painterResource(trailingIcon), null, Modifier.size(20.dp), tint = LocalLelloPalette.current["text-muted"])
        } else {
            Icon(Icons.AutoMirrored.Outlined.KeyboardArrowRight, null, tint = LocalLelloPalette.current["text-muted"])
        }
    }
}

/** A label/value pair for detail screens, with tabular figures for the value. */
@Composable
internal fun DetailRow(label: String, value: @Composable () -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(LelloDimensions.space4),
    ) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium,
            color = LocalLelloPalette.current["text-secondary"])
        value()
    }
}

/** Heading block for lazy lists that cannot nest a LelloSection: 2 dp rule, 20 sp heading, caption. */
@Composable
internal fun ListSectionHeader(title: String, modifier: Modifier = Modifier, meta: String? = null) {
    Column(modifier.fillMaxWidth().padding(top = LelloDimensions.space4, bottom = 4.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        HorizontalDivider(thickness = LelloDimensions.familyRuleWidth, color = LocalLelloPalette.current.familyRule)
        Text(title, Modifier.padding(top = 10.dp).semantics { heading() }, style = MaterialTheme.typography.titleLarge)
        if (meta != null) Text(meta, style = LelloTextStyles.caption)
    }
}

/** Quiet destructive action: a text button in the error color, never a solid fill. */
@Composable
internal fun QuietDangerButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, @DrawableRes icon: Int? = null) {
    TextButton(
        onClick, modifier.heightIn(min = LelloDimensions.controlHeight),
        shape = MaterialTheme.shapes.small,
        colors = ButtonDefaults.textButtonColors(contentColor = LocalLelloPalette.current["error"]),
        contentPadding = if (icon != null) ButtonDefaults.TextButtonWithIconContentPadding else ButtonDefaults.TextButtonContentPadding,
    ) {
        if (icon != null) {
            Icon(painterResource(icon), null, Modifier.size(18.dp))
            Box(Modifier.size(8.dp))
        }
        Text(text)
    }
}

/**
 * App-owned Material dialog with the level 2 shadow and no tonal tint. A [danger] confirmation is
 * the one place a solid error button appears.
 */
@Composable
internal fun CalmDialog(
    title: String,
    onDismiss: () -> Unit,
    confirmLabel: String,
    onConfirm: () -> Unit,
    dismissLabel: String? = null,
    danger: Boolean = false,
    confirmEnabled: Boolean = true,
    dismissEnabled: Boolean = true,
    content: @Composable ColumnScope.() -> Unit,
) {
    val palette = LocalLelloPalette.current
    AlertDialog(
        onDismissRequest = onDismiss,
        modifier = Modifier.lelloShadow(LelloElevation.level2, MaterialTheme.shapes.large),
        shape = MaterialTheme.shapes.large,
        tonalElevation = 0.dp,
        title = { Text(title, style = MaterialTheme.typography.titleLarge) },
        text = { Column(verticalArrangement = Arrangement.spacedBy(12.dp), content = content) },
        confirmButton = {
            if (danger) {
                Button(
                    onConfirm, Modifier.heightIn(min = LelloDimensions.controlHeight), enabled = confirmEnabled,
                    shape = MaterialTheme.shapes.small,
                    colors = ButtonDefaults.buttonColors(containerColor = palette["error-solid"], contentColor = palette["on-error-solid"]),
                ) { Text(confirmLabel) }
            } else {
                LelloTextButton(onClick = onConfirm, enabled = confirmEnabled) { Text(confirmLabel) }
            }
        },
        dismissButton = dismissLabel?.let { label ->
            { LelloTextButton(onClick = onDismiss, enabled = dismissEnabled) { Text(label) } }
        },
    )
}
