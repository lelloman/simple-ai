package com.lelloman.simpleai.ui.navigation

import kotlinx.serialization.Serializable

@Serializable
object Capabilities

@Serializable
object TranslationLanguages

@Serializable
object TranslationTest

@Serializable
object About

@Serializable
object Apps

@Serializable
object Settings

@Serializable
data class ModelDetail(val model: String)
