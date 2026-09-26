# Android LelloDesign adoption

Reference: LelloDesign `e0016d27ce4ccd69bfea2cbec941597859505bf1`, including
`docs/upgrades/open-workspace.md` and the Compose implementation. The newer
open-workspace guide supersedes the older adoption guide's decorative seam.

SimpleAI uses the complete **green** light/dark palettes, bundled Inter fonts,
shared shapes and control wrappers, neutral header divider, and open settings
sections. Appearance is Light/Dark/System, defaults to System, and persists
locally. Wallpaper dynamic colors do not override product identity.

The shared scaffold owns the single route heading and system insets. Phones use
four bottom destinations; detail routes show Back and hide bottom navigation.
At 760 dp, navigation becomes the shared collapsible sidebar. The shared account
shape appears in Settings and at the sidebar foot; its label reflects the actual
gateway sign-in state, without inventing a profile name. App routes, approvals,
downloads, translations and authentication remain app-owned.

The unpublished Compose update is pinned as source in `android/lellodesign` until
it can be replaced with a matching published version. Lazy language lists retain
native scrolling and their own 16 dp gutters. Translation results remain bounded
panels. No connectivity indicator is shown: configuration and sign-in do not
establish an active server connection.

Validation: debug APK and instrumentation APK builds, app lint, 203 app JVM
tests, four shared-palette tests, and two connected model-row accessibility
tests passed. Phone Models was inspected in light and dark; the review prompted
a list-spacing refinement and explicit system-bar icon appearance handling.
The emulator disconnected after accessibility verification; full tablet and
remaining-screen visual acceptance is still pending.

The approved product mark is original H02 (H02-A in the refinement gallery):
two outlined hemispheres with two small folds each. Canonical vector artwork is
`design/simpleai-brain.svg`. Medium Green (`#3ddc84`) identifies the left half;
strong Green (`#006c45`) identifies the right half. The user selected two
components and two artwork tones after exploring the three-tone concepts.
The artwork colors remain fixed across appearance modes. Adaptive launchers use
a white background and centered safe-zone artwork; legacy launchers have density
specific lossless WebP exports. Android themed icons use the same geometry in
monochrome, and the scaffold displays the canonical untinted mark.
