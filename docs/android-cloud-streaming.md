# Android cloud-chat streaming

The Android service can now forward the existing backend's chat-completion SSE
stream instead of waiting for a complete response. No backend deployment is
required by this change.

## Contract

- Existing AIDL methods keep their transaction positions. New clients check
  `capabilities.cloudAi.streaming == true` before using the appended
  `startCloudChat` and `cancelCloudChat` methods, appended after the existing
  `cancelCurrentRequest` method to preserve its transaction ID.
- Each request supplies a unique ID and an `ICloudChatCallback`. The callback
  receives ordered OpenAI-compatible SSE data payloads, ending in `[DONE]` or
  an `{"error":{"message":"..."}}` payload. It must return promptly.
- Cancellation is scoped to the Binder caller UID and request ID. One caller
  cannot cancel another caller's request. There are at most four active streams
  per UID and 64 overall. The shared caller budget further limits work to one
  active request per UID, four overall, and 30 requests per UID per minute.
- Callback-process death or explicit cancellation cancels the job and closes
  its HTTP call, including a blocked read. Service destruction cancels all jobs.
- Only approved client apps may stream. SimpleAI supplies the configured server
  and gateway session; the legacy `authToken` argument is ignored. Requests retain
  HTTPS-only endpoint validation, source-app attribution, and bearer authentication.
  There is a 60-second network inactivity timeout and 180-second total deadline.
  SSE frames are bounded to keep Binder transactions small.
- Clients must assemble indexed tool fragments and validate complete arguments
  before execution. This service forwards payloads, not executable actions.

## Rollout

Ship this SimpleAI Android app together with the streaming-capable
`provider-simpleai` library in client apps. Updating SimpleAI first is safe:
old clients continue using `cloudChat`. Updated clients on old SimpleAI versions
fall back to full-response mode. Do not change the protocol number merely to
enable streaming; capability negotiation is the compatibility boundary.

No builds have been installed or published as part of this implementation.

## Verification

```sh
cd android
./gradlew :app:compileDebugKotlin
./gradlew :app:testDebugUnitTest --tests 'com.lelloman.simpleai.cloud.*'
```

Tests cover SSE framing, oversized/truncated frames, first output before response
completion, and cancellation during a blocked HTTP read. Loopback test URLs are
injected internally; production endpoint validation remains HTTPS-only.

Before rollout, test on-device with both updated apps: incremental text, a tool
turn, clear/logout mid-response, service termination, and an older SimpleAI app
for the fallback path. Automated tests do not replace this Binder/device check.
