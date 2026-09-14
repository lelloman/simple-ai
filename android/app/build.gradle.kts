import java.util.Properties


plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

// Version identity is source-controlled and independent of clone depth or branch history.
val releaseVersion = Properties().apply { rootProject.file("version.properties").inputStream().use { load(it) } }
val appVersionCode = providers.gradleProperty("simpleai.versionCode").orElse(releaseVersion.getProperty("versionCode")).get().toInt()
val appVersionName = providers.gradleProperty("simpleai.versionName").orElse(releaseVersion.getProperty("versionName")).get()
require(appVersionCode in 1..2_100_000_000) { "simpleai.versionCode must be between 1 and 2100000000" }
require(appVersionName.matches(Regex("[0-9]+\\.[0-9]+\\.[0-9]+(?:-[A-Za-z0-9.]+)?"))) { "Use a semantic simpleai.versionName, for example 1.0.227" }

// Load signing.properties for release signing config
val signingProperties = Properties().apply {
    val signingPropsFile = rootProject.file(providers.gradleProperty("simpleai.signingProperties").orElse("signing.properties").get())
    if (signingPropsFile.exists()) {
        signingPropsFile.inputStream().use { load(it) }
    }
}

// Load local.properties for API endpoints
val localProperties = Properties().apply {
    val localPropsFile = rootProject.file("local.properties")
    if (localPropsFile.exists()) {
        localPropsFile.inputStream().use { load(it) }
    }
}

// Cloud LLM endpoint (OpenAI-compatible API)
val cloudLlmEndpoint = localProperties.getProperty("cloud.llm.endpoint", "")

android {
    namespace = "com.lelloman.simpleai"
    compileSdk = 36
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.lelloman.simpleai"
        minSdk = 24
        targetSdk = 36
        versionCode = appVersionCode
        versionName = appVersionName

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }

        // SimpleAI protocol versioning
        buildConfigField("int", "SERVICE_VERSION", "2")
        buildConfigField("int", "MIN_PROTOCOL_VERSION", "2")
        buildConfigField("int", "MAX_PROTOCOL_VERSION", "2")
    }

    buildFeatures {
        compose = true
        aidl = true
        buildConfig = true
    }

    signingConfigs {
        if (signingProperties.containsKey("storeFile")) {
            create("release") {
                storeFile = file(signingProperties.getProperty("storeFile"))
                storePassword = signingProperties.getProperty("storePassword")
                keyAlias = signingProperties.getProperty("keyAlias")
                keyPassword = signingProperties.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        debug {
            buildConfigField("String", "CLOUD_LLM_ENDPOINT", "\"$cloudLlmEndpoint\"")
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            if (signingConfigs.findByName("release") != null) {
                signingConfig = signingConfigs.getByName("release")
            }
            buildConfigField("String", "CLOUD_LLM_ENDPOINT", "\"$cloudLlmEndpoint\"")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

val validateReleaseConfiguration = tasks.register("validateReleaseConfiguration") {
    doLast {
        val required = listOf("storeFile", "storePassword", "keyAlias", "keyPassword")
        check(required.all { !signingProperties.getProperty(it).isNullOrBlank() }) {
            "Release signing is required. Configure storeFile, storePassword, keyAlias and keyPassword in android/signing.properties (or -Psimpleai.signingProperties=/path/to/file)."
        }
        check(file(signingProperties.getProperty("storeFile")).isFile) { "Release keystore file does not exist" }
        logger.lifecycle("Release identity: {} ({})", appVersionName, appVersionCode)
    }
}
tasks.configureEach {
    if (name == "preReleaseBuild") dependsOn(validateReleaseConfiguration)
}

val tokenizerSources = rootProject.file("tokenizer-native")
val tokenizerJni = layout.buildDirectory.dir("generated/tokenizerJni")
val buildTokenizerAndroid = tasks.register<Exec>("buildTokenizerAndroid") {
    workingDir(tokenizerSources)
    inputs.files(fileTree(tokenizerSources) { include("src/**", "Cargo.toml", "Cargo.lock") })
    outputs.dir(tokenizerJni)
    environment("ANDROID_NDK_HOME", android.ndkDirectory.absolutePath)
    environment("RUSTFLAGS", "-C link-arg=-Wl,-z,max-page-size=16384")
    commandLine("cargo", "ndk", "-t", "arm64-v8a", "-t", "armeabi-v7a", "-o", tokenizerJni.get().asFile.absolutePath, "build", "--release", "--locked")
}
android.sourceSets.getByName("main").jniLibs.srcDir(tokenizerJni)
tasks.configureEach {
    if (name.startsWith("merge") && name.endsWith("JniLibFolders")) dependsOn(buildTokenizerAndroid)
}
val buildTokenizerHost = tasks.register<Exec>("buildTokenizerHost") {
    workingDir(tokenizerSources)
    inputs.files(fileTree(tokenizerSources) { include("src/**", "Cargo.toml", "Cargo.lock") })
    outputs.file(File(tokenizerSources, "target/debug/${System.mapLibraryName("simpleai_tokenizer")}"))
    commandLine("cargo", "build", "--locked")
}
tasks.withType<Test>().configureEach {
    dependsOn(buildTokenizerHost)
    inputs.file(buildTokenizerHost.map { it.outputs.files.singleFile })
    systemProperty("java.library.path", File(tokenizerSources, "target/debug").absolutePath)
}

dependencies {
    implementation("androidx.work:work-runtime-ktx:2.10.1")
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.navigation.compose)

    // Networking for model download
    implementation(libs.okhttp)

    // Coroutines
    implementation(libs.coroutines.core)
    implementation(libs.coroutines.android)

    // LLM inference
    implementation(libs.llamacpp.kotlin)

    // ONNX Runtime for transformer models (NLU classification)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")

    // JSON serialization
    implementation(libs.kotlinx.serialization.json)

    // ML Kit Translation and Language ID
    implementation("com.google.mlkit:translate:17.0.3")
    implementation("com.google.mlkit:language-id:17.0.6")

    testImplementation(libs.junit)
    testImplementation(libs.mockk)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.turbine)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}
