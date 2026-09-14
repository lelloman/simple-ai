# Android service lifetime

Clients should bind with `BIND_AUTO_CREATE` and unbind when finished. Opening SimpleAI only binds; it does not start a sticky foreground service. Startup checks download inventory and translation availability without loading native Voice Commands or Local AI engines.

Native models load on first inference. Disk-only models appear as **Downloaded • loads when needed** in the app. For protocol compatibility, service discovery advertises these as `status: ready, loaded: false`; resident models use `loaded: true`. A first request can spend time loading and may report a loading error. After the last inference or download activation completes, a 60-second idle timer releases both native engines. New work cancels that timer; release and work admission are serialized. Model files remain downloaded. Translation clients stay app-owned and can be reused.

Legacy clients that explicitly start the service get a non-sticky foreground service for a 60-second compatibility window, with open-app navigation and active request count. The service then drops its started/foreground state; live bindings continue to keep it available. It is not restarted solely because the process died. Clients must keep their binding while using the API.

Downloads run through WorkManager independently, with progress, pause and open-app notification controls. Their scheduling and foreground lifetime are managed by WorkManager.

JVM tests verify lazy inventory, unload/reload state and idle timer reset. Battery, resident memory and OS background lifecycle behavior still need measurements on supported ARM64 devices; no battery savings figure is claimed.
