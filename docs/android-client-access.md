# Android client access

Service discovery and language inventory are public. Inference and adapter operations require the user's approval in **SimpleAI → Connected apps**. A denied request registers the caller for that panel and returns `CLIENT_NOT_APPROVED`. Retry after approval. SimpleAI's own UID is trusted.

Approval is stored against the complete set of packages and current signing certificates for the calling UID. Apps sharing a UID share trust and budgets. A different signer or changed shared-UID package set requires approval again. Revocation blocks subsequent calls; an operation already running may finish. No prompt text or cloud credentials are stored in the client list.

Each UID can have one active expensive call and start at most 30 calls per 60-second window. Four expensive calls can be active across clients. Excess requests fail immediately with `RATE_LIMITED`; use bounded backoff. Client and budget registries are bounded to 256 entries. Android's Binder transaction limit also applies.

Adapters are internally named `<Binder UID>:<client adapter ID>`. Selecting an adapter and classifying are one serialized transaction. Clearing an adapter only affects the calling UID's active adapter. Clients must supply adapter descriptors again when their adapter is no longer resident.

Device verification still required: bind from two independently signed applications, approve/revoke each, verify denial after signing identity changes, and test cross-client adapter switching under load.
