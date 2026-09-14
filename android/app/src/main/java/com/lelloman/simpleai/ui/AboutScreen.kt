package com.lelloman.simpleai.ui

import com.lelloman.simpleai.R

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.TextButton
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.lelloman.simpleai.BuildConfig

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AboutScreen(
    onBack: () -> Unit
) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    val context = androidx.compose.ui.platform.LocalContext.current
    val uriHandler = androidx.compose.ui.platform.LocalUriHandler.current
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(strings.getString(R.string.ui_about)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = strings.getString(R.string.ui_back)
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(modifier = Modifier.height(16.dp))

            // App icon/logo placeholder
            Text(
                text = "\uD83E\uDD16",
                modifier = Modifier.clearAndSetSemantics {},
                style = MaterialTheme.typography.displayLarge
            )

            Text(
                text = strings.getString(R.string.ui_simpleai),
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = strings.getString(R.string.ui_version, BuildConfig.VERSION_NAME),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Description card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = strings.getString(R.string.ui_about),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = strings.getString(R.string.ui_simpleai_is_a_shared_ai_service_for_compatible_android_apps_use_i),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Features card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = strings.getString(R.string.ui_features),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    FeatureItem("\uD83C\uDFA4", strings.getString(R.string.ui_voice_commands), strings.getString(R.string.ui_understand_commands_supplied_by_a_compatible_app_processing_stays))
                    FeatureItem("\uD83C\uDF10", strings.getString(R.string.ui_translation), strings.getString(R.string.ui_translate_on_this_device_after_downloading_language_packs_try_it_))
                    FeatureItem("\u2601\uFE0F", strings.getString(R.string.ui_cloud_ai), strings.getString(R.string.ui_sends_requests_over_the_internet_to_the_configured_cloud_provider))
                    FeatureItem("\uD83E\uDD16", strings.getString(R.string.ui_local_ai), strings.getString(R.string.ui_generate_text_on_this_device_after_downloading_the_model_no_cloud))
                }
            }

            Text(strings.getString(R.string.ui_getting_connected), style = MaterialTheme.typography.titleMedium)
            Text(strings.getString(R.string.ui_choose_simpleai_in_an_app_that_supports_it_download_the_models_th))
            Text(strings.getString(R.string.ui_check_the_connection), style = MaterialTheme.typography.titleMedium)
            Text(strings.getString(R.string.ui_use_test_on_the_translation_card_to_send_a_translation_through_th))

            Text(strings.getString(R.string.ui_models_and_support), style = MaterialTheme.typography.titleMedium)
            Text(strings.getString(R.string.ui_voice_commands_xlm_roberta_int8, formatSize(com.lelloman.simpleai.model.NluModel.SIZE_BYTES)))
            Text(strings.getString(R.string.ui_local_ai_2, com.lelloman.simpleai.model.LocalAIModel.NAME, formatSize(com.lelloman.simpleai.model.LocalAIModel.SIZE_BYTES)))
            Text(strings.getString(R.string.ui_local_ai_native_inference_requires_arm64_model_downloads_and_the_))
            TextButton(onClick = {
                val clipboard = context.getSystemService(android.content.ClipboardManager::class.java)
                clipboard.setPrimaryClip(android.content.ClipData.newPlainText(strings.getString(R.string.ui_simpleai_diagnostics), supportDiagnostics()))
                android.widget.Toast.makeText(context, strings.getString(R.string.ui_diagnostics_copied), android.widget.Toast.LENGTH_SHORT).show()
            }) { Text(strings.getString(R.string.ui_copy_support_diagnostics)) }
            TextButton(onClick = { uriHandler.openUri("https://github.com/lelloman/simple-ai/issues") }) { Text(strings.getString(R.string.ui_support_and_issue_tracker)) }
            Text(strings.getString(R.string.ui_model_and_library_licenses), style = MaterialTheme.typography.titleMedium)
            Text(strings.getString(R.string.ui_qwen3_apache_2_0_xlm_roberta_mit_onnx_runtime_and_llama_cpp_mit_h))
            TextButton(onClick = { uriHandler.openUri("https://huggingface.co/Qwen/Qwen3-1.7B") }) { Text(strings.getString(R.string.ui_qwen_model_and_license)) }
            TextButton(onClick = { uriHandler.openUri("https://huggingface.co/FacebookAI/xlm-roberta-base") }) { Text(strings.getString(R.string.ui_xlm_roberta_model_and_license)) }
            TextButton(onClick = { uriHandler.openUri("https://developers.google.com/ml-kit/terms") }) { Text(strings.getString(R.string.ui_ml_kit_terms_and_privacy)) }

            // Build info card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = strings.getString(R.string.ui_build_info),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    BuildInfoRow(strings.getString(R.string.ui_version_code), BuildConfig.VERSION_CODE.toString())
                    BuildInfoRow(strings.getString(R.string.ui_build_type), BuildConfig.BUILD_TYPE)
                    BuildInfoRow(strings.getString(R.string.ui_service_version), BuildConfig.SERVICE_VERSION.toString())
                    BuildInfoRow(strings.getString(R.string.ui_protocol_version), strings.getString(R.string.ui_value_3, BuildConfig.MIN_PROTOCOL_VERSION, BuildConfig.MAX_PROTOCOL_VERSION))
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = strings.getString(R.string.ui_made_with_u2764_ufe0f),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun FeatureItem(icon: String, title: String, description: String) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    Column {
        Text(
            text = title,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
        Text(
            text = description,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun BuildInfoRow(label: String, value: String) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    Text(
        text = strings.getString(R.string.ui_value_4, label, value),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}
