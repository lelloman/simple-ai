package com.lelloman.simpleai

import androidx.compose.foundation.layout.WindowInsets
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
            SimpleAITheme {
                val navController = rememberNavController()
                // Share ViewModel across all screens by creating it at NavHost level
                val sharedViewModel: CapabilitiesViewModel = viewModel()

                val entry by navController.currentBackStackEntryAsState()
                val destination = entry?.destination
                val rootScreen = destination == null || destination.hasRoute<Capabilities>() || destination.hasRoute<TranslationTest>() || destination.hasRoute<Apps>() || destination.hasRoute<Settings>()
                Scaffold(contentWindowInsets = WindowInsets(0, 0, 0, 0), bottomBar = {
                    if (rootScreen) NavigationBar {
                        val tabs = listOf(
                            Triple(Capabilities, R.string.nav_models, Icons.AutoMirrored.Filled.List),
                            Triple(TranslationTest, R.string.nav_translate, Icons.Default.Edit),
                            Triple(Apps, R.string.nav_apps, Icons.Default.Person),
                            Triple(Settings, R.string.nav_settings, Icons.Default.Settings)
                        )
                        tabs.forEach { (route, label, icon) ->
                            val selected = when (route) {
                                TranslationTest -> destination?.hasRoute<TranslationTest>() == true
                                Capabilities -> destination?.hasRoute<Capabilities>() != false
                                Apps -> destination?.hasRoute<Apps>() == true
                                else -> destination?.hasRoute<Settings>() == true
                            }
                            NavigationBarItem(selected = selected, onClick = {
                                navController.navigate(route) {
                                    popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }, icon = { Icon(icon, null) }, label = { Text(stringResource(label)) })
                        }
                    }
                }) { padding ->
                    NavHost(navController = navController, startDestination = Capabilities, modifier = Modifier.padding(padding)) {
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
