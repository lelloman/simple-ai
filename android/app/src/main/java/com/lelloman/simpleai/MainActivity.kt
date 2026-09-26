package com.lelloman.simpleai

import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.*
import androidx.compose.ui.unit.dp
import androidx.compose.ui.res.painterResource
import com.lelloman.lellodesign.*
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.navigation.NavDestination.Companion.hasRoute
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.toRoute
import androidx.navigation.compose.currentBackStackEntryAsState
import com.lelloman.simpleai.ui.ConnectedApps
import com.lelloman.simpleai.ui.SettingsScreen
import com.lelloman.simpleai.ui.ModelDetailScreen
import com.lelloman.simpleai.ui.navigation.Apps
import com.lelloman.simpleai.ui.navigation.Settings
import com.lelloman.simpleai.ui.navigation.ModelDetail
import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.lelloman.simpleai.ui.AboutScreen
import com.lelloman.simpleai.ui.CapabilitiesScreen
import com.lelloman.simpleai.ui.CapabilitiesViewModel
import com.lelloman.simpleai.ui.TranslationLanguagesScreen
import com.lelloman.simpleai.ui.TranslationTestScreen
import com.lelloman.simpleai.ui.navigation.About
import com.lelloman.simpleai.ui.navigation.Capabilities
import com.lelloman.simpleai.ui.navigation.TranslationLanguages
import com.lelloman.simpleai.ui.navigation.TranslationTest
import com.lelloman.simpleai.ui.theme.SimpleAITheme

class MainActivity : ComponentActivity() {

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* We proceed regardless of permission result */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestNotificationPermission()
        enableEdgeToEdge()
        setContent {
            val preferences = remember { getSharedPreferences("appearance", MODE_PRIVATE) }
            var appearance by remember { mutableStateOf(LelloAppearance.entries.firstOrNull {
                it.name == preferences.getString("mode", null)
            } ?: LelloAppearance.System) }
            val dark = when (appearance) {
                LelloAppearance.Light -> false
                LelloAppearance.Dark -> true
                LelloAppearance.System -> isSystemInDarkTheme()
            }
            SideEffect {
                androidx.core.view.WindowCompat.getInsetsController(window, window.decorView).apply {
                    isAppearanceLightStatusBars = !dark
                    isAppearanceLightNavigationBars = !dark
                }
            }
            SimpleAITheme(darkTheme = dark) {
                val navController = rememberNavController()
                // Share ViewModel across all screens by creating it at NavHost level
                val sharedViewModel: CapabilitiesViewModel = viewModel()

                val entry by navController.currentBackStackEntryAsState()
                val destination = entry?.destination
                val rootScreen = destination == null || destination.hasRoute<Capabilities>() || destination.hasRoute<TranslationTest>() || destination.hasRoute<Apps>() || destination.hasRoute<Settings>()
                val selectedId = when {
                    destination?.hasRoute<TranslationTest>() == true -> "translate"
                    destination?.hasRoute<Apps>() == true -> "apps"
                    destination?.hasRoute<Settings>() == true || destination?.hasRoute<About>() == true -> "settings"
                    else -> "models"
                }
                val tabs = listOf(
                    LelloDestination("models", stringResource(R.string.nav_models)) { Icon(Icons.AutoMirrored.Filled.List, null) },
                    LelloDestination("translate", stringResource(R.string.nav_translate)) { Icon(Icons.Default.Edit, null) },
                    LelloDestination("apps", stringResource(R.string.nav_apps)) { Icon(Icons.Default.Person, null) },
                    LelloDestination("settings", stringResource(R.string.nav_settings)) { Icon(Icons.Default.Settings, null) },
                )
                val title = when {
                    destination?.hasRoute<TranslationLanguages>() == true -> stringResource(R.string.ui_manage_languages)
                    destination?.hasRoute<About>() == true -> stringResource(R.string.ui_about)
                    destination?.hasRoute<ModelDetail>() == true -> stringResource(
                        if (entry?.toRoute<ModelDetail>()?.model == "voice") R.string.ui_voice_commands else R.string.ui_local_ai)
                    else -> tabs.first { it.id == selectedId }.label
                }
                LelloScaffold(
                    productName = stringResource(R.string.app_name), title = title,
                    destinations = tabs, selectedId = selectedId,
                    mobileNavigation = LelloMobileNavigation.Bottom,
                    bottomDestinations = if (rootScreen) tabs else emptyList(),
                    onBack = if (rootScreen) null else ({ navController.popBackStack(); Unit }),
                    logo = {
                        Surface(color = androidx.compose.ui.graphics.Color.White,
                            shape = androidx.compose.foundation.shape.RoundedCornerShape(7.dp)) {
                            Icon(painterResource(R.drawable.ic_brand), null,
                                modifier = Modifier.size(32.dp), tint = androidx.compose.ui.graphics.Color.Unspecified)
                        }
                    },
                    account = { compact ->
                        val signedIn by sharedViewModel.gatewaySignedIn.collectAsState()
                        LelloAccount(stringResource(if (signedIn) R.string.gateway_signed_in else R.string.gateway_signed_out),
                            onClick = { navController.navigate(Settings) { launchSingleTop = true } }, compact = compact)
                    },
                    actions = { LelloAppearanceSelector(appearance, onSelected = {
                        appearance = it
                        preferences.edit().putString("mode", it.name).apply()
                    }) },
                    onNavigate = { id ->
                        val route = when (id) {
                            "translate" -> TranslationTest
                            "apps" -> Apps
                            "settings" -> Settings
                            else -> Capabilities
                        }
                        navController.navigate(route) {
                            popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                ) { padding ->
                    NavHost(navController = navController, startDestination = Capabilities, modifier = Modifier.padding(padding).consumeWindowInsets(padding)) {
                        composable<Capabilities> {
                            CapabilitiesScreen(sharedViewModel,
                                onNavigateToTranslationLanguages = { navController.navigate(TranslationLanguages) },
                                onOpenModel = { navController.navigate(ModelDetail(it)) })
                        }
                        composable<TranslationTest> {
                            TranslationTestScreen(sharedViewModel, onLanguages = { navController.navigate(TranslationLanguages) })
                        }
                        composable<Apps> { ConnectedApps() }
                        composable<Settings> { SettingsScreen(sharedViewModel, onAbout = { navController.navigate(About) }) }
                        composable<ModelDetail> { detail ->
                            ModelDetailScreen(detail.toRoute<ModelDetail>().model, sharedViewModel) { navController.popBackStack() }
                        }
                        composable<TranslationLanguages> {
                            TranslationLanguagesScreen(sharedViewModel, onBack = { navController.popBackStack() })
                        }
                        composable<About> { AboutScreen(onBack = { navController.popBackStack() }) }
                    }
                }
            }
        }
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }
}
