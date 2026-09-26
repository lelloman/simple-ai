# Pinned LelloDesign Compose snapshot

Source: ../lellodesign/packages/compose/lellodesign at
`e0016d27ce4ccd69bfea2cbec941597859505bf1` (unreleased open-workspace update).
The upstream working changes at adoption were documentation only; Compose source
and generated tokens match this commit. Source, resources, and upstream tests are
vendored here so a clean SimpleAI checkout builds without the sibling repository,
private registry credentials, or an unpublished Maven version.

`GeneratedTokens.kt` is the upstream generated export of its canonical color JSON,
layout CSS and soft-hexagon geometry. Update this snapshot deliberately from an
upstream release; do not maintain separate app-specific edits in library source.
The local Gradle build uses SimpleAI's version catalog and omits publication and
upstream generation tasks. Inter's OFL license is bundled in assets.
