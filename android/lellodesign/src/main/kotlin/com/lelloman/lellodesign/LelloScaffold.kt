package com.lelloman.lellodesign

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

class LelloDestination(val id: String, val label: String, val icon: @Composable () -> Unit)
enum class LelloMobileNavigation { Drawer, Bottom, DrawerAndBottom }

internal val LocalLelloWideLayout = staticCompositionLocalOf { false }

/** Apply inside the scaffold after consuming its system insets; scrolling stays host-owned. */
@Composable
fun LelloWorkspace(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    val wide = LocalLelloWideLayout.current
    Column(modifier.fillMaxWidth().padding(
        horizontal = if (wide) LelloDimensions.workspaceGutter else LelloDimensions.workspaceGutterMobile,
        vertical = if (wide) LelloDimensions.workspaceTop else LelloDimensions.workspaceGutterMobile,
    ), verticalArrangement = Arrangement.spacedBy(
        if (wide) LelloDimensions.workspaceGap else LelloDimensions.workspaceGapMobile,
    ), content = content)
}

data class LelloScaffoldLabels(
    val openNavigation: String = "Open navigation",
    val collapseSidebar: String = "Collapse sidebar",
    val expandSidebar: String = "Expand sidebar",
    val collapsedState: String = "Collapsed",
    val expandedState: String = "Expanded",
    val navigateBack: String = "Back",
)

/**
 * Navigation is controlled by the app, with no routing or authentication dependency.
 * In bottom mode place the account in Settings; the scaffold never puts it in the app bar.
 * DrawerAndBottom uses destinations for the drawer/sidebar and bottomDestinations for tabs.
 * onBack replaces the compact menu/logo with an RTL-aware Back button on detail routes.
 * The host owns the Back stack and system Back handling; an open drawer consumes Back first.
 * Supply drawerState to coordinate host-driven dismissal with navigation.
 * Open/close it from a composition coroutine scope; layout changes close it.
 * The content slot stays at one composition location when window width changes.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LelloScaffold(
    productName: String,
    title: String,
    destinations: List<LelloDestination>,
    selectedId: String,
    onNavigate: (String) -> Unit,
    modifier: Modifier = Modifier,
    mobileNavigation: LelloMobileNavigation = LelloMobileNavigation.Drawer,
    labels: LelloScaffoldLabels = LelloScaffoldLabels(),
    logo: @Composable () -> Unit = {},
    account: @Composable (compact: Boolean) -> Unit = {},
    actions: @Composable RowScope.() -> Unit = {},
    bottomDestinations: List<LelloDestination> = destinations,
    onBack: (() -> Unit)? = null,
    drawerState: DrawerState = rememberDrawerState(DrawerValue.Closed),
    content: @Composable (PaddingValues) -> Unit,
) {
    var collapsed by rememberSaveable { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val latestAccount = rememberUpdatedState(account)
    val movingAccount = remember { movableContentOf<Boolean> { compact -> latestAccount.value(compact) } }
    val palette = LocalLelloPalette.current
    BoxWithConstraints(modifier.fillMaxSize().background(palette["surface"])) {
        val wide = maxWidth >= 760.dp
        val drawerMode = !wide && mobileNavigation != LelloMobileNavigation.Bottom
        val bottomMode = !wide && mobileNavigation != LelloMobileNavigation.Drawer
        val panelWidth by animateDpAsState(if (collapsed) LelloDimensions.sidebarRailWidth else LelloDimensions.sidebarWidth, tween(240), label = "Sidebar width")
        LaunchedEffect(drawerState, wide, mobileNavigation) { drawerState.close() }
        ModalNavigationDrawer(
            drawerState = drawerState,
            gesturesEnabled = drawerMode,
            drawerContent = {
                if (drawerMode) ModalDrawerSheet(drawerState = drawerState, modifier = Modifier.width(minOf(320.dp, maxWidth - 32.dp)), drawerContainerColor = palette["surface-sunken"]) {
                    NavigationPanel(productName, destinations, selectedId,
                        onNavigate = { onNavigate(it); scope.launch { drawerState.close() } },
                        compact = false, dense = false, logo = logo, account = { movingAccount(false) })
                }
            },
        ) {
            Row(Modifier.fillMaxSize()) {
                if (wide) Surface(Modifier.width(panelWidth).fillMaxHeight(), color = palette["surface-sunken"]) {
                    NavigationPanel(productName, destinations, selectedId, onNavigate, collapsed, logo,
                        account = { movingAccount(collapsed) },
                        toggle = {
                            IconButton(onClick = { collapsed = !collapsed }, modifier = Modifier.semantics {
                                stateDescription = if (collapsed) labels.collapsedState else labels.expandedState
                            }) {
                                Icon(if (collapsed) Icons.AutoMirrored.Filled.KeyboardArrowRight else Icons.AutoMirrored.Filled.KeyboardArrowLeft,
                                    if (collapsed) labels.expandSidebar else labels.collapseSidebar)
                            }
                        })
                }
                Scaffold(
                    modifier = Modifier.weight(1f),
                    containerColor = palette["surface"],
                    topBar = {
                        Surface {
                            Box(Modifier.windowInsetsPadding(WindowInsets.statusBars.only(WindowInsetsSides.Top))) {
                                Row(Modifier.fillMaxWidth().heightIn(min = LelloDimensions.headerHeight)
                                    .padding(horizontal = if (wide) LelloDimensions.workspaceGutter else LelloDimensions.workspaceGutterMobile),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                    if (onBack != null) IconButton(onClick = onBack) {
                                        Icon(Icons.AutoMirrored.Filled.ArrowBack, labels.navigateBack)
                                    } else if (drawerMode) IconButton(onClick = { scope.launch { drawerState.open() } }) {
                                        Icon(Icons.Default.Menu, labels.openNavigation)
                                    }
                                    if (!wide && !drawerMode && onBack == null) logo()
                                    Text(title.ifEmpty { productName }, Modifier.weight(1f),
                                        style = if (wide) MaterialTheme.typography.headlineLarge else MaterialTheme.typography.titleLarge)
                                    actions()
                                }
                                HorizontalDivider(Modifier.align(Alignment.BottomCenter), color = palette["border-subtle"])
                            }
                        }
                    },
                    bottomBar = {
                        if (bottomMode && bottomDestinations.isNotEmpty()) {
                            LelloBottomNavigation(bottomDestinations, selectedId, onNavigate)
                        }
                    },
                    content = { padding ->
                        CompositionLocalProvider(LocalLelloWideLayout provides wide) { content(padding) }
                    },
                )
            }
        }
    }
}

@Composable
private fun NavigationPanel(
    productName: String, destinations: List<LelloDestination>, selectedId: String,
    onNavigate: (String) -> Unit, compact: Boolean, logo: @Composable () -> Unit,
    account: @Composable () -> Unit, toggle: (@Composable () -> Unit)? = null,
    dense: Boolean = true,
) {
    val palette = LocalLelloPalette.current
    Column(Modifier.fillMaxSize().background(palette["surface-sunken"])
        .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Vertical))
        .padding(start = if (compact) 8.dp else 12.dp, end = if (compact) 8.dp else 0.dp, bottom = 20.dp)) {
        Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).padding(start = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            if (!compact) Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                logo()
                Text(productName, style = MaterialTheme.typography.titleLarge.copy(fontWeight = androidx.compose.ui.text.font.FontWeight.Bold))
            }
            toggle?.invoke()
        }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            destinations.forEach { item ->
                val selected = selectedId == item.id
                val colors = MaterialTheme.colorScheme
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(topStart = 4.dp, bottomStart = 4.dp))
                    .background(if (selected) colors.surface else palette["surface-sunken"])
                    .lelloSelectionEdge(selected, colors.primary)
                    .selectable(selected = selected, role = Role.Tab, onClick = { onNavigate(item.id) })
                    .semantics(mergeDescendants = true) { contentDescription = item.label }
                    .heightIn(min = 48.dp)
                    .padding(horizontal = 12.dp, vertical = if (dense) 8.dp else 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = if (compact) Arrangement.Center else Arrangement.spacedBy(12.dp)) {
                    CompositionLocalProvider(LocalContentColor provides if (selected) colors.onSurface else colors.onSurfaceVariant) {
                        CompositionLocalProvider(LocalContentColor provides if (selected) colors.primary else colors.onSurfaceVariant) {
                            Box(Modifier.size(20.dp)) { item.icon() }
                        }
                        if (!compact) Text(item.label, style = MaterialTheme.typography.labelLarge,
                            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium)
                    }
                }
            }
        }
        HorizontalDivider()
        Spacer(Modifier.height(12.dp))
        Box(Modifier.fillMaxWidth().padding(end = if (compact) 0.dp else 8.dp), contentAlignment = Alignment.CenterStart) { account() }
    }
}
