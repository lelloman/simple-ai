package com.lelloman.simpleai.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import com.lelloman.lellodesign.LelloTheme

@Composable
fun SimpleAITheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    LelloTheme(product = "green", dark = darkTheme, content = content)
}
