pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // LelloDesign is published to the private Fucina registry (LAN/VPN only).
        // The read token comes from FUCINA_MAVEN_TOKEN, or the workstation's
        // ~/.config/fucina/lellodesign-maven-consume.token when the variable is unset.
        exclusiveContent {
            forRepository {
                maven {
                    name = "FucinaLelloDesign"
                    url = uri("https://fucina.homelab/api/packages/lelloman/maven")
                    credentials {
                        username = "lelloman"
                        password = providers.environmentVariable("FUCINA_MAVEN_TOKEN").orNull
                            ?: File(System.getProperty("user.home"), ".config/fucina/lellodesign-maven-consume.token")
                                .takeIf { it.isFile }?.readText()?.trim()
                    }
                }
            }
            filter { includeModule("com.lelloman", "lellodesign-compose") }
        }
    }
}

rootProject.name = "SimpleAI"
include(":app")
 