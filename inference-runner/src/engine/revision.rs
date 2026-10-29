//! Weight identity is computed during loading, never during inference or status polling.
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

#[derive(Default)]
pub struct Revisions(Mutex<HashMap<String, Option<String>>>);
impl Revisions {
    pub fn get(&self, id: &str) -> Option<String> {
        self.0.lock().unwrap().get(id).cloned().flatten()
    }
    pub fn attempted(&self, id: &str) -> bool {
        self.0.lock().unwrap().contains_key(id)
    }
    pub fn clear(&self, id: &str) {
        self.0.lock().unwrap().remove(id);
    }
    pub async fn load_hf(&self, id: &str) {
        let root = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|p| PathBuf::from(p).join("hub")))
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join(".cache/huggingface/hub")
            });
        let repo = root.join(format!("models--{}", id.replace('/', "--")));
        if Path::new(id).exists() {
            self.load(id, Path::new(id)).await;
        } else if let Ok(commit) = std::fs::read_to_string(repo.join("refs/main")) {
            self.load(id, &repo.join("snapshots").join(commit.trim()))
                .await;
        } else {
            tracing::warn!(
                model = id,
                "No offline HF snapshot found for revision reporting"
            );
        }
    }
    pub async fn load(&self, id: &str, path: &Path) {
        let path = path.to_owned();
        // Remove stale identity before a reload, including failed hashing attempts.
        self.0.lock().unwrap().insert(id.into(), None);
        match tokio::task::spawn_blocking(move || identity(&path)).await {
            Ok(Ok(revision)) => {
                self.0.lock().unwrap().insert(id.into(), Some(revision));
            }
            result => {
                tracing::warn!(model = id, error = ?result, "Unable to identify model weights")
            }
        }
    }
}
fn hash_file(path: &Path) -> std::io::Result<String> {
    let stamp = fingerprint(path)?;
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    if fingerprint(path)? != stamp {
        return Err(std::io::Error::other("weights changed while hashing"));
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn fingerprint(path: &Path) -> std::io::Result<String> {
    let m = std::fs::metadata(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(format!(
            "{}:{}:{}:{}:{}:{}",
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
            m.ino()
        ))
    }
    #[cfg(not(unix))]
    {
        Ok(format!("{}:{:?}", m.len(), m.modified()?))
    }
}
fn identity(path: &Path) -> std::io::Result<String> {
    let path = path.canonicalize()?;
    if path.is_file() {
        let stamp = fingerprint(&path)?;
        let cache = PathBuf::from(format!("{}.simple-ai-sha256", path.display()));
        let digest = hash_file(&path)?;
        if fingerprint(&path)? != stamp {
            return Err(std::io::Error::other("weights changed while hashing"));
        }
        // Atomic sidecar replacement. A read-only model directory does not prevent serving.
        let written = (|| -> std::io::Result<()> {
            let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
            writeln!(file, "{digest}")?;
            file.persist(&cache).map_err(|e| e.error)?;
            Ok(())
        })();
        if let Err(error) = written {
            tracing::warn!(path = %path.display(), %error, "Unable to cache weight SHA-256 beside file");
        }
        return Ok(digest);
    }
    if path
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|p| p == "snapshots")
    {
        if let Some(commit) = path
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Ok(commit.into());
        }
    }
    fn files(
        root: &Path,
        out: &mut Vec<PathBuf>,
        seen: &mut std::collections::HashSet<PathBuf>,
    ) -> std::io::Result<()> {
        if !seen.insert(root.canonicalize()?) {
            return Err(std::io::Error::other(
                "checkpoint directory cycle or duplicate directory symlink",
            ));
        }
        for entry in std::fs::read_dir(root)? {
            let p = entry?.path();
            if p.file_name()
                .is_some_and(|s| s.to_string_lossy().ends_with(".simple-ai-sha256"))
            {
                continue;
            }
            let metadata = std::fs::metadata(&p)?;
            if metadata.is_dir() {
                files(&p, out, seen)?;
            } else if metadata.is_file() {
                out.push(p);
            } else {
                return Err(std::io::Error::other("unsupported checkpoint entry"));
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    files(&path, &mut paths, &mut Default::default())?;
    paths.sort();
    if paths.is_empty() {
        return Err(std::io::Error::other("empty checkpoint directory"));
    }
    let mut hash = Sha256::new();
    for file in paths {
        let name = file
            .strip_prefix(&path)
            .unwrap()
            .as_os_str()
            .as_encoded_bytes();
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name);
        hash.update(hash_file(&file)?.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hashes_weights_and_replaces_sidecar_each_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let first = identity(&path).unwrap();
        assert_eq!(
            first,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(identity(&path).unwrap(), first);
        std::fs::write(&path, b"xyz").unwrap();
        let second = identity(&path).unwrap();
        assert_ne!(second, first);
        assert_eq!(
            std::fs::read_to_string(format!("{}.simple-ai-sha256", path.display()))
                .unwrap()
                .trim(),
            second
        );
    }
    #[test]
    fn directory_identity_tracks_content_and_names() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), b"weights").unwrap();
        let a = identity(dir.path()).unwrap();
        assert_eq!(identity(dir.path()).unwrap(), a);
        std::fs::rename(dir.path().join("a"), dir.path().join("b")).unwrap();
        assert_ne!(identity(dir.path()).unwrap(), a);
    }
    #[tokio::test]
    async fn failed_reload_clears_identity_and_can_retry() {
        let revisions = Revisions::default();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.gguf");
        std::fs::write(&path, b"weights").unwrap();
        revisions.load("model", &path).await;
        assert!(revisions.get("model").is_some());
        std::fs::remove_file(&path).unwrap();
        revisions.load("model", &path).await;
        assert_eq!(revisions.get("model"), None);
        assert!(revisions.attempted("model"));
        std::fs::write(&path, b"replacement").unwrap();
        revisions.load("model", &path).await;
        assert!(revisions.get("model").is_some());
    }
    #[test]
    fn hf_snapshot_uses_commit() {
        let dir = tempfile::tempdir().unwrap();
        let commit = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3";
        let snapshot = dir.path().join("snapshots").join(commit);
        std::fs::create_dir_all(&snapshot).unwrap();
        assert_eq!(identity(&snapshot).unwrap(), commit);
    }
    #[cfg(unix)]
    #[test]
    fn checkpoint_directory_cycle_fails_without_recursing_forever() {
        let dir = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("cycle")).unwrap();
        assert!(identity(dir.path())
            .unwrap_err()
            .to_string()
            .contains("cycle"));
    }
}
