package com.lelloman.lellodesign

import androidx.compose.ui.graphics.Color
import org.junit.Assert.*
import org.junit.Test

class PaletteTest {
    @Test fun allPalettesResolveEveryMaterialAndStatusRole() {
        assertEquals(22, LelloPalettes.names.size)
        for (name in LelloPalettes.names) {
            val palette = LelloPalettes.named(name)
            val scheme = palette.materialColorScheme()
            assertEquals(palette["primary"], scheme.primary)
            assertEquals(palette["text"], scheme.onSurface)
            assertEquals(palette["error-solid"], scheme.error)
            assertEquals(if (name.endsWith("-dark")) 179f / 255f else 153f / 255f, palette["scrim"].alpha, 0.001f)
            for (tone in listOf("success", "warning", "error", "info")) {
                assertNotEquals(Color.Unspecified, palette["$tone-container"])
                assertNotEquals(Color.Unspecified, palette["on-$tone-container"])
            }
        }
    }
    @Test fun independentPrimaryPreservesFamilyColorsInBothAppearances() {
        for (family in listOf("blue", "green")) for (dark in listOf(false, true)) {
            val original = LelloPalettes.forProduct(family, dark)
            val purple = LelloPalettes.forProduct("purple$family", dark)
            assertNotEquals(original["primary"], purple["primary"])
            assertEquals(original["secondary"], purple["secondary"])
            assertEquals(original["tertiary"], purple["tertiary"])
        }
    }
    @Test fun customPaletteDoesNotMutateSharedDefaults() {
        val original = LelloPalettes.forProduct()
        val custom = original.withColors(overrides = mapOf("primary" to Color.Magenta))
        assertEquals(Color.Magenta, custom["primary"])
        assertNotEquals(custom["primary"], original["primary"])
    }
    @Test(expected = IllegalStateException::class) fun unknownPaletteFailsClearly() {
        LelloPalettes.named("missing")
    }
}
