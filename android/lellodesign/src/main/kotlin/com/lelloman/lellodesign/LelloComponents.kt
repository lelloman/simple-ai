package com.lelloman.lellodesign

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.vector.PathParser
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.*
import java.util.Locale

/** The canonical 40 × 40 soft hexagon used by the web account control. */
val LelloAccountShape: Shape = object : Shape {
    override fun createOutline(size: Size, layoutDirection: LayoutDirection, density: Density): Outline {
        val path = PathParser().parsePathString(AccountPath).toPath()
        path.transform(Matrix().apply { scale(size.width / 40f, size.height / 40f) })
        return Outline.Generic(path)
    }
}

/** Compatibility API. Open workspaces use a neutral divider, without a decorative seam. */
@Deprecated("Use HorizontalDivider with the Lello border-subtle color; the decorative seam is retired.")
@Composable
@Suppress("UNUSED_PARAMETER")
fun LelloSeam(modifier: Modifier = Modifier, gap: Dp = LelloDimensions.space2) {
    HorizontalDivider(modifier, color = LocalLelloPalette.current["border-subtle"])
}

/** Draw inside the selected target without changing geometry; follows the logical end in RTL. */
internal fun Modifier.lelloSelectionEdge(selected: Boolean, color: Color, top: Boolean = false): Modifier = drawWithContent {
    drawContent()
    if (selected) {
        val width = LelloDimensions.familySelectionWidth.toPx()
        if (top) drawLine(color, Offset(0f, width / 2), Offset(size.width, width / 2), width)
        else {
            val x = if (layoutDirection == LayoutDirection.Ltr) size.width - width / 2 else width / 2
            drawLine(color, Offset(x, 0f), Offset(x, size.height), width)
        }
    }
}

@Composable
fun LelloAvatar(name: String, modifier: Modifier = Modifier, portrait: (@Composable () -> Unit)? = null) {
    val initials = remember(name) {
        name.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.take(2)
            .joinToString("") { String(Character.toChars(it.codePointAt(0))) }
            .uppercase(Locale.ROOT).ifEmpty { "?" }
    }
    Box(modifier.size(36.dp).background(MaterialTheme.colorScheme.primary, LelloAccountShape)
        .clearAndSetSemantics {}, contentAlignment = Alignment.Center) {
        Box(Modifier.size(22.dp).clip(CircleShape).background(MaterialTheme.colorScheme.surface),
            contentAlignment = Alignment.Center) {
            if (portrait != null) portrait() else Text(initials, style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface)
        }
    }
}

@Composable
fun LelloAccount(
    name: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    subtitle: String? = null,
    compact: Boolean = false,
    accessibilityLabel: String = "Account: $name",
    portrait: (@Composable () -> Unit)? = null,
) {
    Surface(onClick = onClick, modifier = modifier.then(if (compact) Modifier else Modifier.fillMaxWidth()).heightIn(min = 56.dp)
        .semantics { contentDescription = accessibilityLabel },
        shape = MaterialTheme.shapes.small, color = Color.Transparent) {
        Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            LelloAvatar(name, portrait = portrait)
            if (!compact) Column {
                Text(name, style = MaterialTheme.typography.titleSmall)
                if (subtitle != null) Text(subtitle, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

/** Material's button defaults use a pill; this wrapper applies the Lello control radius. */
@Composable
fun LelloButton(onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true,
                content: @Composable RowScope.() -> Unit) {
    Button(onClick, modifier.heightIn(min = LelloDimensions.controlHeight), enabled = enabled,
        shape = MaterialTheme.shapes.small, content = content)
}

@Composable
fun LelloTextField(value: String, onValueChange: (String) -> Unit, label: @Composable () -> Unit,
                   modifier: Modifier = Modifier, enabled: Boolean = true, isError: Boolean = false,
                   supportingText: (@Composable () -> Unit)? = null, singleLine: Boolean = true,
                   suffix: (@Composable () -> Unit)? = null, keyboardOptions: KeyboardOptions = KeyboardOptions.Default) {
    OutlinedTextField(value, onValueChange, modifier, enabled = enabled, label = label,
        isError = isError, supportingText = supportingText, singleLine = singleLine,
        shape = MaterialTheme.shapes.small, suffix = suffix, keyboardOptions = keyboardOptions)
}

@Composable
fun LelloCard(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Surface(modifier, shape = MaterialTheme.shapes.medium,
        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
        Column(Modifier.padding(LelloDimensions.space4), verticalArrangement = Arrangement.spacedBy(12.dp), content = content)
    }
}

enum class LelloTone { Info, Success, Warning, Error }

@Composable
fun LelloAlert(title: String, modifier: Modifier = Modifier, tone: LelloTone = LelloTone.Info,
               content: @Composable ColumnScope.() -> Unit = {}) {
    val palette = LocalLelloPalette.current
    val key = tone.name.lowercase(Locale.ROOT)
    Surface(modifier.semantics { liveRegion = if (tone == LelloTone.Error) LiveRegionMode.Assertive else LiveRegionMode.Polite },
        shape = MaterialTheme.shapes.extraSmall,
        color = palette["$key-container"], contentColor = palette["on-$key-container"],
        border = androidx.compose.foundation.BorderStroke(1.dp, palette["$key-border"])) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            content()
        }
    }
}

/** Product supplies the meaningful icon and action; identity comes from shared treatment. */
@Composable
fun LelloState(title: String, description: String, modifier: Modifier = Modifier,
               icon: @Composable () -> Unit = {}, action: @Composable () -> Unit = {}) {
    Column(modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        CompositionLocalProvider(LocalContentColor provides MaterialTheme.colorScheme.primary) { icon() }
        Text(title, style = MaterialTheme.typography.titleLarge)
        Text(description, style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        action()
    }
}

/** Secondary action with the same geometry and target size as LelloButton. */
@Composable
fun LelloOutlinedButton(onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true,
    contentPadding: PaddingValues = ButtonDefaults.ContentPadding, content: @Composable RowScope.() -> Unit) {
    OutlinedButton(onClick, modifier.heightIn(min = LelloDimensions.controlHeight), enabled = enabled,
        shape = MaterialTheme.shapes.small, contentPadding = contentPadding, content = content)
}

@Composable
fun LelloTextButton(onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit) {
    TextButton(onClick, modifier.heightIn(min = LelloDimensions.controlHeight), enabled = enabled,
        shape = MaterialTheme.shapes.small, content = content)
}

@Composable
fun LelloFilterChip(selected: Boolean, onClick: () -> Unit, label: @Composable () -> Unit,
    modifier: Modifier = Modifier, enabled: Boolean = true, leadingIcon: (@Composable () -> Unit)? = null) {
    FilterChip(selected, onClick, label, modifier, enabled = enabled, leadingIcon = leadingIcon,
        shape = MaterialTheme.shapes.small)
}

/** An open section with a strong structural rule; content remains product-owned. */
@Composable
fun LelloSection(title: String, modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        HorizontalDivider(thickness = LelloDimensions.familyRuleWidth, color = MaterialTheme.colorScheme.onSurface)
        Text(title, style = MaterialTheme.typography.titleLarge, modifier = Modifier.semantics { heading() })
        Column(verticalArrangement = Arrangement.spacedBy(24.dp), content = content)
    }
}

@Composable
fun LelloSettingsSection(title: String, modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) =
    LelloSection(title, modifier, content)

/** Standard Material 3 mobile navigation, using the current Lello theme. */
@Composable
fun LelloBottomNavigation(destinations: List<LelloDestination>, selectedId: String,
    onNavigate: (String) -> Unit, modifier: Modifier = Modifier) {
    NavigationBar(modifier = modifier) {
        destinations.forEach { item ->
            NavigationBarItem(
                selected = item.id == selectedId,
                onClick = { onNavigate(item.id) },
                icon = item.icon,
                label = { Text(item.label) },
            )
        }
    }
}

/** Decorative palette preview; the accompanying choice supplies its accessible name. */
@Composable
fun LelloPaletteSwatch(colors: List<Color>, modifier: Modifier = Modifier) {
    Row(modifier.size(width = 26.dp, height = 18.dp).clip(MaterialTheme.shapes.extraSmall)
        .border(1.dp, MaterialTheme.colorScheme.outlineVariant, MaterialTheme.shapes.extraSmall).clearAndSetSemantics {}) {
        colors.forEach { color -> Spacer(Modifier.weight(1f).fillMaxHeight().background(color)) }
    }
}

/** Non-interactive progress for a finite sequence. Completed steps survive host-owned state changes. */
@Composable
fun LelloStepProgress(completed: Int, total: Int, modifier: Modifier = Modifier, current: Int? = null) {
    require(total > 0) { "total must be positive" }
    val count = completed.coerceIn(0, total)
    Row(modifier.fillMaxWidth().semantics {
        progressBarRangeInfo = ProgressBarRangeInfo(count.toFloat(), 0f..total.toFloat(), (total - 1).coerceAtLeast(0))
    }, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        repeat(total) { index ->
            Spacer(Modifier.weight(1f).height(5.dp).clip(MaterialTheme.shapes.extraSmall)
                .background(when {
                    index < count -> MaterialTheme.colorScheme.primary
                    index == current -> MaterialTheme.colorScheme.secondary
                    else -> MaterialTheme.colorScheme.outlineVariant
                }))
        }
    }
}
