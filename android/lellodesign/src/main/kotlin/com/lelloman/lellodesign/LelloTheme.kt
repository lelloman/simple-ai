package com.lelloman.lellodesign

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/** A complete semantic palette. Custom palettes must retain every canonical role. */
@Immutable
class LelloPalette(val name: String, colors: Map<String, Color>) {
    val colors: Map<String, Color> = colors.toMap()
    operator fun get(role: String): Color = colors[role]
        ?: error("Missing Lello color role: $role")

    /** Override complete role groups, including readable foregrounds and containers. */
    fun withColors(name: String = this.name, overrides: Map<String, Color>): LelloPalette =
        LelloPalette(name, colors + overrides)
}

object LelloPalettes {
    val names: List<String> get() = GeneratedPalettes.keys.toList()
    fun named(name: String): LelloPalette = GeneratedPalettes[name]
        ?: error("Unknown Lello palette '$name'. Available: ${names.joinToString()}")
    fun forProduct(product: String = "blue", dark: Boolean = false): LelloPalette =
        named("$product-${if (dark) "dark" else "light"}")
}

/** All Material roles are assigned explicitly to avoid fallback purple/default surfaces. */
fun LelloPalette.materialColorScheme(): ColorScheme = lightColorScheme(
    primary = this["primary"], onPrimary = this["on-primary"],
    primaryContainer = this["primary-container"], onPrimaryContainer = this["on-primary-container"],
    inversePrimary = this["icon-light"],
    secondary = this["secondary"], onSecondary = this["on-secondary"],
    secondaryContainer = this["secondary-container"], onSecondaryContainer = this["on-secondary-container"],
    tertiary = this["tertiary"], onTertiary = this["on-tertiary"],
    tertiaryContainer = this["tertiary-container"], onTertiaryContainer = this["on-tertiary-container"],
    background = this["background"], onBackground = this["text"],
    surface = this["surface"], onSurface = this["text"],
    surfaceVariant = this["surface-sunken"], onSurfaceVariant = this["text-secondary"],
    surfaceTint = this["primary"], inverseSurface = this["surface-inverse"], inverseOnSurface = this["text-inverse"],
    error = this["error-solid"], onError = this["on-error-solid"],
    errorContainer = this["error-container"], onErrorContainer = this["on-error-container"],
    outline = this["border-control"], outlineVariant = this["border-subtle"], scrim = this["scrim"],
    surfaceBright = this["surface-raised"], surfaceDim = this["surface-sunken"],
    surfaceContainerLowest = this["background"], surfaceContainerLow = this["surface"],
    surfaceContainer = this["surface"], surfaceContainerHigh = this["surface-raised"],
    surfaceContainerHighest = this["surface-pressed"],
    primaryFixed = this["primary-container"], primaryFixedDim = this["primary"],
    onPrimaryFixed = this["on-primary-container"], onPrimaryFixedVariant = this["on-primary-container"],
    secondaryFixed = this["secondary-container"], secondaryFixedDim = this["secondary"],
    onSecondaryFixed = this["on-secondary-container"], onSecondaryFixedVariant = this["on-secondary-container"],
    tertiaryFixed = this["tertiary-container"], tertiaryFixedDim = this["tertiary"],
    onTertiaryFixed = this["on-tertiary-container"], onTertiaryFixedVariant = this["on-tertiary-container"],
)

val LelloShapes = Shapes(
    extraSmall = RoundedCornerShape(LelloDimensions.radiusSmall),
    small = RoundedCornerShape(LelloDimensions.radiusControl),
    medium = RoundedCornerShape(LelloDimensions.radiusPanel),
    large = RoundedCornerShape(LelloDimensions.radiusDialog),
    extraLarge = RoundedCornerShape(LelloDimensions.radiusDialog),
)

/** Bundled static faces work offline, including Android versions before variable fonts. */
val LelloFontFamily = FontFamily(
    Font(R.font.lello_inter_regular, FontWeight.Normal),
    Font(R.font.lello_inter_medium, FontWeight.Medium),
    Font(R.font.lello_inter_semibold, FontWeight.SemiBold),
    Font(R.font.lello_inter_bold, FontWeight.Bold),
)

private fun text(size: Int, height: Int, weight: FontWeight = FontWeight.Normal) = TextStyle(
    fontFamily = LelloFontFamily, fontSize = size.sp, lineHeight = height.sp, fontWeight = weight,
)
val LelloTypography = Typography(
    displayLarge = text(57, 64), displayMedium = text(45, 52), displaySmall = text(36, 44),
    headlineLarge = text(24, 32, FontWeight.SemiBold).copy(fontSize = LelloDimensions.fontTitle, lineHeight = LelloDimensions.lineTitle), headlineMedium = text(20, 28, FontWeight.SemiBold),
    headlineSmall = text(18, 28, FontWeight.SemiBold), titleLarge = text(18, 28, FontWeight.SemiBold).copy(fontSize = LelloDimensions.fontSection, lineHeight = LelloDimensions.lineSection),
    titleMedium = text(16, 24, FontWeight.SemiBold), titleSmall = text(14, 20, FontWeight.SemiBold),
    bodyLarge = text(16, 24).copy(fontSize = LelloDimensions.fontBody, lineHeight = LelloDimensions.lineBody), bodyMedium = text(14, 20), bodySmall = text(12, 16),
    labelLarge = text(14, 20, FontWeight.Medium).copy(fontSize = LelloDimensions.fontLabel, lineHeight = LelloDimensions.lineLabel), labelMedium = text(12, 16, FontWeight.Medium),
    labelSmall = text(11, 16, FontWeight.Medium),
)

val LocalLelloPalette = staticCompositionLocalOf { LelloPalettes.forProduct() }

/** Theme standard Material 3 components and expose extended Lello status/interaction roles. */
@Composable
fun LelloTheme(
    product: String = "blue",
    dark: Boolean = isSystemInDarkTheme(),
    palette: LelloPalette = LelloPalettes.forProduct(product, dark),
    typography: Typography = LelloTypography,
    content: @Composable () -> Unit,
) {
    val colors = remember(palette) { palette.materialColorScheme() }
    CompositionLocalProvider(LocalLelloPalette provides palette) {
        MaterialTheme(colorScheme = colors, typography = typography, shapes = LelloShapes, content = content)
    }
}
