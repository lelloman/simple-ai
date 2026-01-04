package com.lelloman.simpleai

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

                NavHost(navController = navController, startDestination = Capabilities) {
                    composable<Capabilities> {
                        CapabilitiesScreen(
                            viewModel = sharedViewModel,
                            onNavigateToTranslationLanguages = { navController.navigate(TranslationLanguages) },
                            onNavigateToTranslationTest = { navController.navigate(TranslationTest) },
                            onNavigateToAbout = { navController.navigate(About) }
                        )
                    }
                    composable<TranslationLanguages> {
                        TranslationLanguagesScreen(
                            viewModel = sharedViewModel,
                            onBack = { navController.popBackStack() }
                        )
                    }
                    composable<TranslationTest> {
                        TranslationTestScreen(
                            viewModel = sharedViewModel,
                            onBack = { navController.popBackStack() }
                        )
                    }
                    composable<About> {
                        AboutScreen(
                            onBack = { navController.popBackStack() }
                        )
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
