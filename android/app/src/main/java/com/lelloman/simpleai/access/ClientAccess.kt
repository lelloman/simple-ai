package com.lelloman.simpleai.access

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Process
import java.security.MessageDigest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Approval is tied to package names and signing certificates, never a reusable UID alone. */
class ClientAccess private constructor(private val context: Context) {
    data class Client(val identity: String, val packages: String, val approved: Boolean)
    private val preferences = context.getSharedPreferences("client_access", Context.MODE_PRIVATE)
    private val _clients = MutableStateFlow(readClients())
    val clients = _clients.asStateFlow()
    private fun readClients() = preferences.all.entries.mapNotNull { (key, value) ->
        (value as? String)?.let { Client(key, it.substringAfter('|'), it.startsWith("yes|")) }
    }.sortedBy { it.packages }

    @Synchronized fun allowed(uid: Int): Boolean {
        if (uid == Process.myUid()) return true
        val packages = context.packageManager.getPackagesForUid(uid)?.sorted() ?: return false
        if (packages.isEmpty()) return false
        val identity = try {
            packages.joinToString(";") { name ->
                @Suppress("DEPRECATION")
                val info = context.packageManager.getPackageInfo(name, if (Build.VERSION.SDK_INT >= 28) PackageManager.GET_SIGNING_CERTIFICATES else PackageManager.GET_SIGNATURES)
                @Suppress("DEPRECATION")
                val signatures = if (Build.VERSION.SDK_INT >= 28) info.signingInfo?.apkContentsSigners else info.signatures
                require(!signatures.isNullOrEmpty())
                name + ":" + signatures.map { signature ->
                    MessageDigest.getInstance("SHA-256").digest(signature.toByteArray()).joinToString("") { "%02x".format(it) }
                }.sorted().joinToString(",")
            }
        } catch (_: Exception) { return false }
        val existing = preferences.getString(identity, null)
        if (existing == null && preferences.all.size < 256) {
            preferences.edit().putString(identity, "no|${packages.joinToString(", ")}").apply()
            _clients.value = readClients()
        }
        return existing?.startsWith("yes|") == true
    }
    @Synchronized fun setApproved(client: Client, approved: Boolean) {
        preferences.edit().putString(client.identity, "${if (approved) "yes" else "no"}|${client.packages}").apply()
        _clients.value = readClients()
    }
    companion object {
        @Volatile private var instance: ClientAccess? = null
        fun get(context: Context): ClientAccess = instance ?: synchronized(this) {
            instance ?: ClientAccess(context.applicationContext).also { instance = it }
        }
    }
}
