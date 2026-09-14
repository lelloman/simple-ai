use jni::{
    objects::{JObject, JString},
    sys::{jlong, jstring},
    JNIEnv,
};
use std::{
    collections::HashMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc, Mutex, OnceLock,
    },
};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

static NEXT_ID: AtomicI64 = AtomicI64::new(1);
static TOKENIZERS: OnceLock<Mutex<HashMap<i64, Arc<Tokenizer>>>> = OnceLock::new();
fn registry() -> &'static Mutex<HashMap<i64, Arc<Tokenizer>>> {
    TOKENIZERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err("Tokenizer rejected the input".into()))
}

fn load(json: &str, max_length: usize) -> Result<Tokenizer, String> {
    if !(2..=4096).contains(&max_length) {
        return Err("Adapter max_length must be between 2 and 4096".into());
    }
    let mut t = Tokenizer::from_bytes(json.as_bytes()).map_err(|e| e.to_string())?;
    t.with_truncation(Some(TruncationParams {
        max_length,
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    t.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(max_length),
        pad_id: 1,
        pad_token: "<pad>".into(),
        ..Default::default()
    }));
    Ok(t)
}

fn encode(t: &Tokenizer, text: &str) -> Result<String, String> {
    let encoded = t
        .encode_char_offsets(text, true)
        .map_err(|e| e.to_string())?;
    let mut utf16 = vec![0usize];
    for c in text.chars() {
        utf16.push(utf16.last().unwrap() + c.len_utf16());
    }
    let offsets: Vec<_> = encoded
        .get_offsets()
        .iter()
        .map(|&(a, b)| (utf16[a], utf16[b]))
        .collect();
    Ok(serde_json::json!({"ids": encoded.get_ids(), "mask": encoded.get_attention_mask(), "offsets": offsets}).to_string())
}

#[no_mangle]
pub extern "system" fn Java_com_lelloman_simpleai_nlu_NativeTokenizer_create(
    mut env: JNIEnv,
    _: JObject,
    json: JString,
    max_length: i32,
) -> jlong {
    let result = guarded(|| {
        let json: String = env.get_string(&json).map_err(|e| e.to_string())?.into();
        let t = load(&json, max_length as usize)?;
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        registry()
            .lock()
            .map_err(|e| e.to_string())?
            .insert(id, Arc::new(t));
        Ok::<_, String>(id)
    });
    match result {
        Ok(id) => id,
        Err(e) => {
            let _ = env.throw_new("java/lang/IllegalArgumentException", e);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lelloman_simpleai_nlu_NativeTokenizer_encodeNative(
    mut env: JNIEnv,
    _: JObject,
    id: jlong,
    text: JString,
) -> jstring {
    let result = guarded(|| {
        let text: String = env.get_string(&text).map_err(|e| e.to_string())?.into();
        let t = registry()
            .lock()
            .map_err(|e| e.to_string())?
            .get(&id)
            .cloned()
            .ok_or("Tokenizer is closed")?;
        let encoded = encode(&t, &text)?;
        env.new_string(encoded)
            .map(|s| s.into_raw())
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(s) => s,
        Err(e) => {
            let _ = env.throw_new("java/lang/IllegalStateException", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_lelloman_simpleai_nlu_NativeTokenizer_destroy(
    _: JNIEnv,
    _: JObject,
    id: jlong,
) {
    if let Ok(mut entries) = registry().lock() {
        entries.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_tokenizer_and_sequence_length() {
        assert!(load("{}", 64).is_err());
        assert!(load("{}", 0).is_err());
    }
}
