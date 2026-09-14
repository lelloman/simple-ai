package com.lelloman.simpleai.cloud

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import com.lelloman.simpleai.model.ModelRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

/** AppAuth handles browser callbacks, state/nonce and PKCE; credentials stay in SimpleAI. */
class GatewayLoginActivity : ComponentActivity() {
    private val auth by lazy { ModelRepository.get(this).gatewayAuth }
    private var launched = false
    private val login = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        lifecycleScope.launch {
            try { result.data?.let { auth.complete(it) } ?: auth.reportFailure() }
            catch (e: CancellationException) { throw e }
            catch (_: Exception) { auth.reportFailure() }
            finally { finish() }
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        launched = savedInstanceState?.getBoolean("launched") ?: false
        if (!launched) lifecycleScope.launch {
            try { val intent = auth.begin(); launched = true; login.launch(intent) }
            catch (e: CancellationException) { throw e }
            catch (_: Exception) { auth.reportFailure(); finish() }
        }
    }
    override fun onSaveInstanceState(outState: Bundle) { outState.putBoolean("launched", launched); super.onSaveInstanceState(outState) }
}
