# Android LelloDesign adoption

Reference: the published `com.lelloman:lellodesign-compose:0.5.0` from Fucina's
Maven registry, following LelloDesign's `docs/upgrades/0.5.0.md`, design
principles and screen specifications. It replaced the source snapshot of
LelloDesign `e0016d27` that was vendored in `android/lellodesign` while the
open-workspace update was unpublished.

Building needs Fucina access (LAN or household VPN, private DNS, a JVM that
trusts its CA) and a read token: `FUCINA_MAVEN_TOKEN`, or the workstation file
`~/.config/fucina/lellodesign-maven-consume.token` when the variable is unset.
The token is a build credential only; it never reaches the APK.

SimpleAI uses the **green** light/dark palettes, the bundled Inter fonts,
shared shapes and control wrappers, and open sections. Appearance is
Light/Dark/System in **Settings → Appearance**, defaults to System, and
persists locally; the header carries no controls. Wallpaper dynamic colors do
not override product identity.

The shared scaffold owns the single route heading and system insets. Phones use
four bottom destinations with Material Symbols Outlined glyphs; detail routes
show Back and hide bottom navigation. At 760 dp navigation becomes the shared
collapsible sidebar. The product symbol sits in `LelloProductMark` (on-primary
on the primary hexagon, the two hemispheres at .78 and full opacity). The
shared account shape appears in Settings and at the sidebar foot; its label
reflects the actual gateway sign-in state, without inventing a profile name.
App routes, approvals, downloads, translations and authentication remain
app-owned.

Screen composition follows the 0.5.0 calm rules: one type scale from
`LelloTypography`/`LelloTextStyles` (20 sp section headings, 16 sp row titles,
14 sp secondary text, 12 sp captions, tabular figures for sizes), open sections
with row dividers instead of cards, statuses as soft pills (container fill,
dot, no outline), one primary action per step, quiet destructive actions (an
error-coloured text button) with the solid error button only inside the
confirmation dialog, and app dialogs at elevation level 2 without tonal tint.
Empty states use `LelloState` with the honeycomb illustration. Lazy language
lists keep native scrolling and their own 16 dp gutters. No connectivity
indicator is shown: configuration and sign-in do not establish an active server
connection.

Validation for the 0.5.0 redesign: `./scripts/check` (app JVM tests, lint,
debug and instrumentation APK builds, tokenizer tests and 16 KB alignment)
passed. Every route was reviewed in light and dark through Robolectric captures
with demo data; tablet width and on-device rendering were not re-checked.

The approved product mark is original H02 (H02-A in the refinement gallery):
two outlined hemispheres with two small folds each. Canonical vector artwork is
`design/simpleai-brain.svg`. Medium Green (`#3ddc84`) identifies the left half;
strong Green (`#006c45`) identifies the right half. The user selected two
components and two artwork tones after exploring the three-tone concepts.
The artwork colors remain fixed across appearance modes. Adaptive launchers use
a white background and centered safe-zone artwork; legacy launchers have density
specific lossless WebP exports. Android themed icons use the same geometry in
monochrome. In the app, the scaffold shows the same geometry as a single-colour
symbol (`ic_brand_symbol`) inside `LelloProductMark`.
