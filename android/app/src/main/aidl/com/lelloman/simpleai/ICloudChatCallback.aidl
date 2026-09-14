package com.lelloman.simpleai;

// Ordered, bounded OpenAI-compatible SSE data payloads; "[DONE]" is terminal.
// Synchronous callback applies backpressure; implementations must return promptly.
interface ICloudChatCallback {
    void onEvent(String data);
}
