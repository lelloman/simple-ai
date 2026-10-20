# Apps screen: before and after

Representative screenshots for [LLPR/AI-52](https://crumbles.lelloman.com/w/LLPR/AI/52).
Fixture clients and metadata are synthetic; the screenshots are rendered from the
real Compose screens by Robolectric native-graphics tests, not hand-drawn.

- `before-*`: `ConnectedApps.kt` at `master` (`cee591e`).
- `after-*`: this branch's `ConnectedApps.kt`.

The matrix covers empty, pending, approved and mixed lists in light and dark
themes, plus many-entry scrolling and long labels at 320 dp width with a 2x font
scale. To regenerate the after screenshots:

```bash
cd android
./gradlew :app:testDebugUnitTest --tests 'com.lelloman.simpleai.ui.ConnectedApps*'
```

Images land in `android/app/build/screenshots/ai-52/`.
