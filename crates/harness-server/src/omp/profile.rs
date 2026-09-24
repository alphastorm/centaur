use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};

use crate::omp::protocol::protocol_error;
use crate::{HarnessServerError, Result};

pub(crate) const PROFILE_ID: &str = "centaur-safe";
pub(crate) const OMP_VERSION: &str = "18.3.0";
pub(crate) const SAFE_TOOLS: &[&str] = &[
    "bash", "edit", "glob", "grep", "lsp", "read", "task", "todo", "write",
];

const APPROVED_ENV: &[&str] = &[
    "HOME",
    "PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
];

#[derive(Debug, Clone)]
pub(crate) struct OmpProfile {
    allowed_models: BTreeSet<(String, String)>,
    binary: PathBuf,
    session_root: PathBuf,
}

impl OmpProfile {
    pub(crate) fn load() -> Result<Self> {
        if env::var("CENTAUR_OMP_ENABLED").as_deref() != Ok("1") {
            return Err(protocol_error(
                "OMP harness is disabled; the operator must set CENTAUR_OMP_ENABLED=1",
            ));
        }
        let binary = env::var_os("CENTAUR_OMP_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("omp"));
        let binary = if binary.components().count() > 1 {
            let binary = binary.canonicalize().map_err(|error| {
                HarnessServerError::Protocol(format!(
                    "OMP profile {PROFILE_ID} binary is unavailable: {error}"
                ))
            })?;
            validate_executable(&binary)?;
            binary
        } else {
            binary
        };
        verify_binary_version(&binary)?;
        Self::with_binary(binary)
    }

    fn with_binary(binary: PathBuf) -> Result<Self> {
        let allowed_models = parse_allowed_models()?;
        let session_root = env::var_os("CENTAUR_OMP_SESSION_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/home/agent/.omp/centaur-sessions"));
        if !session_root.is_absolute() {
            return Err(protocol_error("OMP session root must be absolute"));
        }
        if let Ok(metadata) = fs::symlink_metadata(&session_root)
            && metadata.file_type().is_symlink()
        {
            return Err(protocol_error("OMP session root may not be a symlink"));
        }
        fs::create_dir_all(&session_root)?;
        #[cfg(unix)]
        fs::set_permissions(&session_root, fs::Permissions::from_mode(0o700))?;
        let session_root = session_root.canonicalize()?;
        let mappings = session_root.join("mappings");
        fs::create_dir_all(&mappings)?;
        #[cfg(unix)]
        fs::set_permissions(&mappings, fs::Permissions::from_mode(0o700))?;
        Ok(Self {
            allowed_models,
            binary,
            session_root,
        })
    }

    pub(crate) fn command(&self, cwd: &Path) -> Command {
        let mut command = Command::new(&self.binary);
        command.current_dir(cwd);
        command.env_clear();
        for name in APPROVED_ENV {
            if let Some(value) = env::var_os(name) {
                command.env(name, value);
            }
        }
        command.env("PI_CODING_AGENT_DIR", &self.session_root);
        command.args([
            "--mode",
            "rpc",
            "--no-extensions",
            "--no-skills",
            "--no-rules",
            "--no-title",
            "--session-dir",
        ]);
        command.arg(&self.session_root);
        command.args(["--tools", &SAFE_TOOLS.join(",")]);
        command
    }

    pub(crate) fn verify_allowed_model(&self, provider: &str, model: &str) -> Result<()> {
        if !self
            .allowed_models
            .contains(&(provider.to_owned(), model.to_owned()))
        {
            return Err(protocol_error(format!(
                "OMP provider/model {provider}/{model} is not in the operator allowlist"
            )));
        }
        Ok(())
    }

    pub(crate) fn session_root(&self) -> &Path {
        &self.session_root
    }

    pub(crate) fn mapping_dir(&self) -> PathBuf {
        self.session_root.join("mappings")
    }

    pub(crate) fn verify_state(&self, state: &serde_json::Value) -> Result<()> {
        let steering = state
            .get("steeringMode")
            .and_then(serde_json::Value::as_str);
        let follow_up = state
            .get("followUpMode")
            .and_then(serde_json::Value::as_str);
        let interrupt = state
            .get("interruptMode")
            .and_then(serde_json::Value::as_str);
        if steering != Some("one-at-a-time")
            || follow_up != Some("one-at-a-time")
            || interrupt != Some("immediate")
        {
            return Err(protocol_error(
                "OMP effective queue modes do not match centaur-safe",
            ));
        }
        let tools = state
            .get("dumpTools")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| protocol_error("OMP get_state omitted dumpTools"))?
            .iter()
            .map(|tool| {
                tool.get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| protocol_error("OMP dumpTools entry omitted name"))
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let expected = SAFE_TOOLS
            .iter()
            .map(|name| (*name).to_string())
            .collect::<BTreeSet<_>>();
        if tools != expected {
            return Err(protocol_error(format!(
                "OMP effective tools differ from centaur-safe: expected {expected:?}, observed {tools:?}"
            )));
        }
        let session_file = state
            .get("sessionFile")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| protocol_error("OMP get_state omitted sessionFile"))?;
        validate_contained_session_path(&self.session_root, Path::new(session_file))?;
        Ok(())
    }
}

pub(crate) fn validate_contained_session_path(root: &Path, path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err(protocol_error("OMP session path must be absolute"));
    }
    let normalized = normalize_without_symlinks(path)?;
    if !normalized.starts_with(root) {
        return Err(protocol_error(
            "OMP session path escapes the operator session root",
        ));
    }
    if path.exists() {
        let canonical = path.canonicalize()?;
        if !canonical.starts_with(root) {
            return Err(protocol_error(
                "OMP session path resolves outside the operator session root",
            ));
        }
        return Ok(canonical);
    }
    let parent = path
        .parent()
        .ok_or_else(|| protocol_error("OMP session path has no parent"))?;
    let canonical_parent = parent.canonicalize()?;
    if !canonical_parent.starts_with(root) {
        return Err(protocol_error(
            "OMP session parent resolves outside the operator session root",
        ));
    }
    Ok(path.to_path_buf())
}

fn normalize_without_symlinks(path: &Path) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(Path::new("/")),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(protocol_error("OMP session path escapes its root"));
                }
            }
            std::path::Component::Normal(value) => normalized.push(value),
        }
    }
    Ok(normalized)
}

fn parse_allowed_models() -> Result<BTreeSet<(String, String)>> {
    let raw = env::var("CENTAUR_OMP_ALLOWED_MODELS").map_err(|_| {
        protocol_error("CENTAUR_OMP_ALLOWED_MODELS is required when OMP is enabled")
    })?;
    parse_allowed_model_list(&raw)
}

fn parse_allowed_model_list(raw: &str) -> Result<BTreeSet<(String, String)>> {
    let mut models = BTreeSet::new();
    for entry in raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (provider, model) = entry.split_once('/').ok_or_else(|| {
            protocol_error("OMP allowlist entries must use provider/model syntax")
        })?;
        if provider.is_empty()
            || model.is_empty()
            || provider.chars().any(char::is_whitespace)
            || model.chars().any(char::is_whitespace)
        {
            return Err(protocol_error(
                "OMP allowlist contains an invalid provider/model",
            ));
        }
        models.insert((provider.to_owned(), model.to_owned()));
    }
    if models.is_empty() {
        return Err(protocol_error(
            "OMP provider/model allowlist may not be empty",
        ));
    }
    Ok(models)
}

fn verify_binary_version(binary: &Path) -> Result<()> {
    let output = Command::new(binary)
        .arg("--version")
        .env_clear()
        .envs(
            APPROVED_ENV
                .iter()
                .filter_map(|name| env::var_os(name).map(|value| (*name, value))),
        )
        .output()
        .map_err(|error| protocol_error(format!("failed to execute OMP version check: {error}")))?;
    let observed = String::from_utf8_lossy(&output.stdout);
    let expected = format!("omp/{OMP_VERSION}");
    if !output.status.success() || observed.trim() != expected {
        return Err(protocol_error(format!(
            "OMP binary version mismatch: expected {expected}"
        )));
    }
    Ok(())
}

fn validate_executable(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(protocol_error("OMP profile binary is not a regular file"));
    }
    #[cfg(unix)]
    if metadata.mode() & 0o111 == 0 {
        return Err(protocol_error("OMP profile binary is not executable"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{parse_allowed_model_list, validate_contained_session_path};

    #[test]
    fn rejects_session_path_escape() {
        let root = std::env::temp_dir().join(format!("omp-profile-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let error = validate_contained_session_path(&root, &root.join("../escape.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("escapes"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn parses_exact_operator_model_allowlist() {
        let models =
            parse_allowed_model_list("anthropic/claude-sonnet-4-5,openrouter/anthropic/claude")
                .unwrap();
        assert!(models.contains(&("anthropic".to_owned(), "claude-sonnet-4-5".to_owned())));
        assert!(models.contains(&("openrouter".to_owned(), "anthropic/claude".to_owned())));
        assert!(parse_allowed_model_list("anthropic").is_err());
        assert!(parse_allowed_model_list("anthropic/claude sonnet").is_err());
        assert!(parse_allowed_model_list("").is_err());
    }
}
