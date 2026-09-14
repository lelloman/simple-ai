# Cloud diagnostics

Cloud diagnostics use the same metadata-only policy in debug and release builds. Logs include message/tool counts, HTTP status, response character counts and failure class. They exclude endpoint URLs, authorization headers, cache keys, prompts, answers, tool arguments, raw HTTP error bodies and exception messages/stack traces that may embed response snippets.

HTTP error bodies are neither consumed for logging nor echoed to clients. Client errors contain a fixed category and, where useful, numeric HTTP status. The response is always closed, including rejected responses. No content logging toggle is provided.

To investigate an error, record app version, time, HTTP status and failure category. Share conversation content separately only when the user chooses to provide it.
