# Accessibility verification

Implemented contextual capability/language actions, silent decorative icons, one ready announcement, a named swap action, stacked language selectors, wrapping card actions and a growing translation input. Compose instrumentation checks a 280 dp model list at 200% font scale and model/language navigation actions.

Run `./gradlew connectedDebugAndroidTest` on a connected Android device. Manual acceptance checks remain required (no device was connected during implementation):

1. Enable TalkBack. Navigate Models → Languages, Translate, Apps and Settings → About. Confirm each language download/delete and app approval names its target, swap announces its purpose, and status/decorations are not read twice.
2. Repeat at largest system font and display size, portrait, landscape and split screen. Verify full labels, usable scroll, accessible dropdown choices, and no overlapping controls.
3. Download a language; verify progress and errors are announced sensibly. Enter multiline text and inspect both result and error at large text size.
4. Check light/dark themes with Accessibility Scanner for contrast and touch targets. Do not infer a measured contrast score from source review.

The instrumentation source/build is an automated check, not evidence that TalkBack or physical-device scenarios have passed.
