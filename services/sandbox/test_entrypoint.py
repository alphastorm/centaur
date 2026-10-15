from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ENTRYPOINT = Path(__file__).with_name("entrypoint.sh")


class EntrypointOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.agent = self.home / ".omp" / "agent"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        # Unrelated image-only helpers are not needed for configuration startup.
        for name in ("install-tool-shims", "compose-system-prompt"):
            helper = self.bin_dir / name
            helper.write_text("#!/bin/sh\nexit 0\n")
            helper.chmod(0o755)
        self.harness = self.root / "harness"
        (self.harness / "codex").mkdir(parents=True)
        (self.harness / "codex" / "config.toml").write_text('model = "default"\n')
        (self.harness / "claude").mkdir()
        (self.harness / "claude" / "settings.json").write_text("{}\n")

    def run_entrypoint(self, **extra_env: str) -> subprocess.CompletedProcess[str]:
        # Do not inherit credentials, tool paths, or provider settings from HOME.
        return subprocess.run(
            ["bash", str(ENTRYPOINT), "sh", "-c", "printf 'harness-started\\n'"],
            cwd=self.root,
            env={
                "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
                "HOME": str(self.home),
                "CENTAUR_HARNESS_CONFIG_DIR": str(self.harness),
                "CENTAUR_STATE_DIR": str(self.root / "absent-state"),
                "CENTAUR_TOOLS_AUTO_RELOAD": "false",
                "GOOGLE_APPLICATION_CREDENTIALS": str(self.root / "unused-adc.json"),
                **extra_env,
            },
            text=True,
            capture_output=True,
            timeout=30,
        )

    def write_base(self, filename: str, content: str) -> Path:
        self.agent.mkdir(parents=True, exist_ok=True)
        path = self.agent / filename
        path.write_text(content)
        return path

    def assert_started(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "harness-started\n")
        self.assertTrue((self.home / ".ready").is_file())

    def assert_rejected(
        self, result: subprocess.CompletedProcess[str], name: str
    ) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(name, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.home / ".ready").exists())

    def test_missing_overlays_do_not_create_omp_state(self) -> None:
        self.assert_started(self.run_entrypoint())
        self.assertFalse((self.home / ".omp").exists())

    def test_missing_and_blank_overlays_preserve_existing_files(self) -> None:
        config = self.write_base("config.yml", "# keep comment\ntask:\n  batch: false\n")
        models = self.write_base("models.yml", "providers: {}\n")
        before = {path: (path.read_bytes(), path.stat()) for path in (config, models)}
        self.assert_started(
            self.run_entrypoint(OMP_CONFIG_OVERLAY=" \n", OMP_MODELS_OVERLAY="")
        )
        for path, (content, metadata) in before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, metadata.st_mtime_ns)
            self.assertEqual(path.stat().st_mode, metadata.st_mode)

    def test_config_yaml_deep_merge_preserves_keys_and_replaces_lists(self) -> None:
        config = self.write_base(
            "config.yml",
            "task:\n  batch: false\n  maxConcurrency: 4\n"
            "advisor:\n  enabled: false\n  reviewInterval: 3\n"
            "disabledProviders: [apple, ollama]\n",
        )
        self.assert_started(
            self.run_entrypoint(
                OMP_CONFIG_OVERLAY=(
                    "task:\n  batch: true\n  structured: true\n"
                    "advisor:\n  enabled: true\n  syncBacklog: '1'\n"
                    "disabledProviders: [apple]\n"
                )
            )
        )
        self.assertEqual(
            json.loads(config.read_text()),
            {
                "task": {"batch": True, "maxConcurrency": 4, "structured": True},
                "advisor": {"enabled": True, "reviewInterval": 3, "syncBacklog": "1"},
                "disabledProviders": ["apple"],
            },
        )

    def test_models_json_deep_merge_keeps_other_providers_and_token_limits(self) -> None:
        models = self.write_base(
            "models.yml",
            'providers:\n  anthropic:\n    baseUrl: https://old.example\n'
            '    headers:\n      X-Deployment: sandbox\n  openai:\n    models: []\n',
        )
        self.assert_started(
            self.run_entrypoint(
                OMP_MODELS_OVERLAY=json.dumps(
                    {
                        "providers": {
                            "anthropic": {
                                "baseUrl": "https://models.example",
                                "headers": {"X-Route": "bounded"},
                                "models": [{"id": "example", "maxTokens": 8192}],
                            }
                        }
                    }
                )
            )
        )
        providers = json.loads(models.read_text())["providers"]
        self.assertEqual(providers["openai"], {"models": []})
        self.assertEqual(providers["anthropic"]["baseUrl"], "https://models.example")
        self.assertEqual(
            providers["anthropic"]["headers"],
            {"X-Deployment": "sandbox", "X-Route": "bounded"},
        )
        self.assertEqual(providers["anthropic"]["models"], [{"id": "example", "maxTokens": 8192}])

    def test_both_overlays_create_private_files_and_start_healthy(self) -> None:
        env = {
            "OMP_CONFIG_OVERLAY": '{"task":{"structured":true}}',
            "OMP_MODELS_OVERLAY": "providers:\n  anthropic:\n    baseUrl: https://models.example\n",
        }
        self.assert_started(self.run_entrypoint(**env))
        for directory in (self.agent.parent, self.agent):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for filename in ("config.yml", "models.yml"):
            metadata = (self.agent / filename).stat()
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_uid, os.getuid())
            self.assertEqual(metadata.st_gid, os.getgid())
        before = (self.agent / "config.yml").read_bytes()
        self.assert_started(self.run_entrypoint(**env))
        self.assertEqual((self.agent / "config.yml").read_bytes(), before)

    def test_existing_file_permissions_owner_and_inode_are_preserved(self) -> None:
        config = self.write_base("config.yml", "task: {batch: false}\n")
        config.chmod(0o640)
        before = config.stat()
        self.assert_started(self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {batch: true}"))
        after = config.stat()
        self.assertEqual((after.st_mode, after.st_uid, after.st_gid, after.st_ino),
                         (before.st_mode, before.st_uid, before.st_gid, before.st_ino))

    def test_yaml_fallback_and_legacy_models_json_are_merged_in_place(self) -> None:
        config = self.write_base("config.yaml", "task: {batch: false, structured: true}\n")
        models = self.write_base("models.json", '{"providers":{"openai":{"models":[]}}}\n')
        self.assert_started(
            self.run_entrypoint(
                OMP_CONFIG_OVERLAY="task: {batch: true}",
                OMP_MODELS_OVERLAY="providers: {anthropic: {baseUrl: 'https://models.example'}}",
            )
        )
        self.assertEqual(json.loads(config.read_text())["task"], {"batch": True, "structured": True})
        self.assertIn("openai", json.loads(models.read_text())["providers"])
        self.assertFalse((self.agent / "config.yml").exists())
        self.assertFalse((self.agent / "models.yml").exists())

    def test_canonical_files_win_over_yaml_fallbacks(self) -> None:
        config = self.write_base("config.yml", "task: {structured: true}\n")
        fallback = self.write_base("config.yaml", "task: {structured: false}\n")
        self.assert_started(self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {batch: true}"))
        self.assertTrue(json.loads(config.read_text())["task"]["structured"])
        self.assertEqual(fallback.read_text(), "task: {structured: false}\n")

    def test_malformed_overlays_fail_before_readiness_without_echoing_input(self) -> None:
        for name in ("OMP_CONFIG_OVERLAY", "OMP_MODELS_OVERLAY"):
            with self.subTest(name=name):
                result = self.run_entrypoint(**{name: 'sensitive-marker: ["unterminated'})
                self.assert_rejected(result, name)
                self.assertNotIn("sensitive-marker", result.stderr)
        self.assertFalse((self.home / ".omp").exists())

    def test_non_object_overlays_are_rejected(self) -> None:
        for raw in ("null", "false", "[one, two]", '"text"'):
            with self.subTest(raw=raw):
                self.assert_rejected(
                    self.run_entrypoint(OMP_CONFIG_OVERLAY=raw), "OMP_CONFIG_OVERLAY"
                )

    def test_credential_and_auth_keys_are_rejected_at_every_depth(self) -> None:
        for key in (
            "token", "access_token", "refreshToken", "accessTokens", "refresh_tokens",
            "apiKey", "API_KEY", "clientSecret", "clientsecretvalue", "password",
            "session-cookie", "Cookie2", "Authorization",
            "auth", "oauth", "auth.broker.url", "X-Api-Key",
        ):
            for name in ("OMP_CONFIG_OVERLAY", "OMP_MODELS_OVERLAY"):
                with self.subTest(key=key, name=name):
                    raw = json.dumps({"providers": {"example": {"models": [{key: "sensitive-marker"}]}}})
                    result = self.run_entrypoint(**{name: raw})
                    self.assert_rejected(result, name)
                    self.assertNotIn("sensitive-marker", result.stderr)
        self.assertFalse((self.home / ".omp").exists())

    def test_invalid_second_overlay_does_not_modify_first_file(self) -> None:
        config = self.write_base("config.yml", "task: {batch: false}\n")
        result = self.run_entrypoint(
            OMP_CONFIG_OVERLAY="task: {batch: true}", OMP_MODELS_OVERLAY="providers: ["
        )
        self.assert_rejected(result, "OMP_MODELS_OVERLAY")
        self.assertEqual(config.read_text(), "task: {batch: false}\n")

    def test_malformed_second_base_does_not_modify_first_file(self) -> None:
        config = self.write_base("config.yml", "task: {batch: false}\n")
        models = self.write_base("models.yml", "providers: [\n")
        result = self.run_entrypoint(
            OMP_CONFIG_OVERLAY="task: {batch: true}", OMP_MODELS_OVERLAY="providers: {}"
        )
        self.assert_rejected(result, "OMP_MODELS_OVERLAY")
        self.assertEqual(config.read_text(), "task: {batch: false}\n")
        self.assertEqual(models.read_text(), "providers: [\n")

    def test_malformed_base_is_not_overwritten(self) -> None:
        config = self.write_base("config.yml", "task: [\n")
        self.assert_rejected(
            self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {batch: true}"), "OMP_CONFIG_OVERLAY"
        )
        self.assertEqual(config.read_text(), "task: [\n")

    def test_symlinked_targets_and_fallbacks_cannot_escape(self) -> None:
        outside = self.root / "outside.yml"
        outside.write_text("unchanged: true\n")
        self.agent.mkdir(parents=True)
        for filename, name in (
            ("config.yml", "OMP_CONFIG_OVERLAY"),
            ("config.yaml", "OMP_CONFIG_OVERLAY"),
            ("models.yml", "OMP_MODELS_OVERLAY"),
            ("models.yaml", "OMP_MODELS_OVERLAY"),
            ("models.json", "OMP_MODELS_OVERLAY"),
        ):
            with self.subTest(filename=filename):
                target = self.agent / filename
                target.symlink_to("../../../outside.yml")
                self.assert_rejected(self.run_entrypoint(**{name: "task: {}"}), name)
                self.assertTrue(target.is_symlink())
                target.unlink()
        self.assertEqual(outside.read_text(), "unchanged: true\n")
        target.symlink_to(outside)
        self.assert_rejected(
            self.run_entrypoint(OMP_MODELS_OVERLAY="providers: {}"), "OMP_MODELS_OVERLAY"
        )
        self.assertEqual(outside.read_text(), "unchanged: true\n")

    def test_dangling_symlink_target_is_not_replaced(self) -> None:
        self.agent.mkdir(parents=True)
        target = self.agent / "config.yml"
        outside = self.root / "not-created.yml"
        target.symlink_to(outside)
        self.assert_rejected(
            self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {}"), "OMP_CONFIG_OVERLAY"
        )
        self.assertTrue(target.is_symlink())
        self.assertFalse(outside.exists())

    def test_symlinked_agent_directory_is_refused(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        self.agent.parent.mkdir()
        self.agent.symlink_to(outside, target_is_directory=True)
        self.assert_rejected(
            self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {}"), "OMP_CONFIG_OVERLAY"
        )
        self.assertEqual(list(outside.iterdir()), [])

    def test_unmanaged_symlinked_omp_root_is_refused(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        self.agent.parent.symlink_to(outside, target_is_directory=True)
        self.assert_rejected(
            self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {}"), "OMP_CONFIG_OVERLAY"
        )
        self.assertEqual(list(outside.iterdir()), [])

    def test_hardlinked_file_cannot_modify_an_outside_inode(self) -> None:
        outside = self.root / "outside.yml"
        outside.write_text("task: {}\n")
        self.agent.mkdir(parents=True)
        os.link(outside, self.agent / "config.yml")
        self.assert_rejected(
            self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {batch: true}"), "OMP_CONFIG_OVERLAY"
        )
        self.assertEqual(outside.read_text(), "task: {}\n")

    def test_readonly_configuration_is_not_replaced(self) -> None:
        config = self.write_base("config.yml", "task: {}\n")
        config.chmod(0o444)
        before = config.stat()
        self.assert_rejected(
            self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {batch: true}"), "OMP_CONFIG_OVERLAY"
        )
        self.assertEqual(config.read_text(), "task: {}\n")
        self.assertEqual(config.stat().st_mode, before.st_mode)
        self.assertEqual(config.stat().st_ino, before.st_ino)

    def test_redirected_config_paths_and_profiles_are_refused(self) -> None:
        for key, value in (
            ("PI_CONFIG_DIR", "../outside"),
            ("PI_CODING_AGENT_DIR", str(self.root / "outside")),
            ("PI_CODING_AGENT_DIR", "../../outside"),
            ("OMP_PROFILE", "../../outside"),
            ("PI_PROFILE", "other-profile"),
        ):
            with self.subTest(key=key, value=value):
                self.assert_rejected(
                    self.run_entrypoint(OMP_CONFIG_OVERLAY="task: {}", **{key: value}),
                    "OMP_CONFIG_OVERLAY",
                )
        self.assertFalse((self.root / "outside").exists())

    def test_persistent_state_link_is_supported_and_preserves_file_metadata(self) -> None:
        state = self.root / "state"
        persistent_agent = state / "omp" / "agent"
        persistent_agent.mkdir(parents=True)
        config = persistent_agent / "config.yml"
        config.write_text("task: {structured: true}\n")
        config.chmod(0o640)
        before = config.stat()
        self.assert_started(
            self.run_entrypoint(
                CENTAUR_STATE_DIR=str(state), OMP_CONFIG_OVERLAY="task: {batch: true}"
            )
        )
        self.assertTrue((self.home / ".omp").is_symlink())
        self.assertEqual(self.agent.resolve(), persistent_agent.resolve())
        self.assertEqual(json.loads(config.read_text())["task"], {"structured": True, "batch": True})
        after = config.stat()
        self.assertEqual((after.st_mode, after.st_uid, after.st_gid, after.st_ino),
                         (before.st_mode, before.st_uid, before.st_gid, before.st_ino))

    def test_persistent_state_omp_target_cannot_be_a_symlink(self) -> None:
        state = self.root / "state"
        state.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (state / "omp").symlink_to(outside, target_is_directory=True)
        self.assert_rejected(
            self.run_entrypoint(
                CENTAUR_STATE_DIR=str(state), OMP_CONFIG_OVERLAY="task: {batch: true}"
            ),
            "OMP_CONFIG_OVERLAY",
        )
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
