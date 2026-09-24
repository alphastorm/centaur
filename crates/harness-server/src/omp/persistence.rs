use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::Result;
use crate::omp::profile::{OMP_VERSION, OmpProfile, PROFILE_ID, validate_contained_session_path};
use crate::omp::protocol::protocol_error;

const MAPPING_SCHEMA: u32 = 1;
const MAX_MAPPING_BYTES: u64 = 32 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SessionMapping {
    schema_version: u32,
    thread_id: String,
    profile_id: String,
    omp_version: String,
    pub(crate) session_id: String,
    pub(crate) session_file: PathBuf,
}

impl SessionMapping {
    pub(crate) fn new(thread_id: &str, session_id: String, session_file: PathBuf) -> Self {
        Self {
            schema_version: MAPPING_SCHEMA,
            thread_id: thread_id.to_string(),
            profile_id: PROFILE_ID.to_string(),
            omp_version: OMP_VERSION.to_string(),
            session_id,
            session_file,
        }
    }
}

pub(crate) fn save(profile: &OmpProfile, mapping: &SessionMapping) -> Result<()> {
    validate_mapping(profile, &mapping.thread_id, mapping)?;
    let path = mapping_path(profile, &mapping.thread_id);
    reject_symlink_if_present(&path)?;
    let temporary = profile.mapping_dir().join(format!(
        ".{}.{}.tmp",
        mapping_filename(&mapping.thread_id),
        Uuid::new_v4()
    ));
    let bytes = serde_json::to_vec(mapping)?;
    if bytes.len() as u64 > MAX_MAPPING_BYTES {
        return Err(protocol_error("OMP session mapping exceeds its size limit"));
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(&temporary)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    #[cfg(unix)]
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    File::open(profile.mapping_dir())?.sync_all()?;
    Ok(())
}

pub(crate) fn load(profile: &OmpProfile, thread_id: &str) -> Result<Option<SessionMapping>> {
    let path = mapping_path(profile, thread_id);
    reject_symlink_if_present(&path)?;
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() > MAX_MAPPING_BYTES {
        return Err(protocol_error(
            "OMP resume mapping is not a bounded regular file",
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(&path)?
        .take(MAX_MAPPING_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MAPPING_BYTES {
        return Err(protocol_error("OMP resume mapping exceeds its size limit"));
    }
    let mapping: SessionMapping = serde_json::from_slice(&bytes)
        .map_err(|error| protocol_error(format!("OMP resume mapping is invalid: {error}")))?;
    validate_mapping(profile, thread_id, &mapping)?;
    Ok(Some(mapping))
}

fn validate_mapping(
    profile: &OmpProfile,
    expected_thread_id: &str,
    mapping: &SessionMapping,
) -> Result<()> {
    if mapping.schema_version != MAPPING_SCHEMA
        || mapping.thread_id != expected_thread_id
        || mapping.profile_id != PROFILE_ID
        || mapping.omp_version != OMP_VERSION
        || mapping.session_id.trim().is_empty()
    {
        return Err(protocol_error(
            "OMP resume mapping does not match thread, profile, or runtime version",
        ));
    }
    validate_contained_session_path(profile.session_root(), &mapping.session_file)?;
    Ok(())
}

fn mapping_path(profile: &OmpProfile, thread_id: &str) -> PathBuf {
    profile.mapping_dir().join(mapping_filename(thread_id))
}

fn mapping_filename(thread_id: &str) -> String {
    format!("{:x}.json", Sha256::digest(thread_id.as_bytes()))
}

fn reject_symlink_if_present(path: &Path) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path)
        && metadata.file_type().is_symlink()
    {
        return Err(protocol_error("OMP session mapping may not be a symlink"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_thread_keys_with_shared_prefix_resume_distinct_sessions() {
        let root = std::env::temp_dir().join(format!("omp-long-key-{}", Uuid::new_v4()));
        let profile = OmpProfile::for_test(PathBuf::from("unused"), &root);
        let session_file = profile.session_root().join("session.jsonl");
        fs::write(&session_file, b"{}").unwrap();
        let first = format!("{}{}", "x".repeat(256), "a".repeat(256));
        let second = format!("{}{}", "x".repeat(256), "b".repeat(256));
        let result = (|| -> Result<()> {
            save(
                &profile,
                &SessionMapping::new(&first, "first".into(), session_file.clone()),
            )?;
            save(
                &profile,
                &SessionMapping::new(&second, "second".into(), session_file),
            )?;
            let first_mapping = load(&profile, &first)?.unwrap();
            let second_mapping = load(&profile, &second)?.unwrap();
            assert_eq!(
                (first_mapping.thread_id, first_mapping.session_id),
                (first, "first".into())
            );
            assert_eq!(
                (second_mapping.thread_id, second_mapping.session_id),
                (second, "second".into())
            );
            Ok(())
        })();
        fs::remove_dir_all(root).unwrap();
        result.unwrap();
    }

    #[test]
    fn mapping_filename_uses_the_entire_thread_key() {
        let prefix = "x".repeat(256);
        assert_ne!(
            mapping_filename(&format!("{prefix}a")),
            mapping_filename(&format!("{prefix}b"))
        );
    }

    #[test]
    fn load_rejects_mismatched_full_thread_key() {
        let root = std::env::temp_dir().join(format!("omp-key-identity-{}", Uuid::new_v4()));
        let profile = OmpProfile::for_test(PathBuf::from("unused"), &root);
        let session_file = profile.session_root().join("session.jsonl");
        fs::write(&session_file, b"{}").unwrap();
        let mapping = SessionMapping::new("other", "session".into(), session_file);
        fs::write(
            mapping_path(&profile, "requested"),
            serde_json::to_vec(&mapping).unwrap(),
        )
        .unwrap();
        let result = load(&profile, "requested");
        fs::remove_dir_all(root).unwrap();
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not match thread")
        );
    }

    #[test]
    fn mapping_filename_cannot_escape_directory() {
        let filename = mapping_filename("../../other/thread\\name");
        assert!(!filename.contains('/'));
        assert!(!filename.contains('\\'));
        assert!(filename.ends_with(".json"));
    }
    #[cfg(unix)]
    #[test]
    fn rejects_symlink_mapping_file() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("omp-mapping-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.json");
        let mapping = root.join("mapping.json");
        std::fs::write(&target, b"{}").unwrap();
        symlink(&target, &mapping).unwrap();
        assert!(
            reject_symlink_if_present(&mapping)
                .unwrap_err()
                .to_string()
                .contains("symlink")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
