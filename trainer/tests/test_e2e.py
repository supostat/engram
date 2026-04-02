import json
import subprocess
import sys


class TestTrainerSubprocessJsonLines:
    def test_every_output_line_is_valid_json_with_type_field(
        self, training_database_path, tmp_path,
    ):
        models_path = str(tmp_path / "models")
        completed = subprocess.run(
            [
                sys.executable, "-m", "engram_trainer",
                "--database", training_database_path,
                "--models-path", models_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        stdout_lines = [
            line for line in completed.stdout.strip().split("\n")
            if line.strip()
        ]
        assert len(stdout_lines) >= 2

        for line in stdout_lines:
            parsed = json.loads(line)
            assert "type" in parsed, f"Missing 'type' field in: {line}"


class TestTrainerSubprocessCompleteMessage:
    def test_last_line_is_complete_type(
        self, training_database_path, tmp_path,
    ):
        models_path = str(tmp_path / "models")
        completed = subprocess.run(
            [
                sys.executable, "-m", "engram_trainer",
                "--database", training_database_path,
                "--models-path", models_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        stdout_lines = [
            line for line in completed.stdout.strip().split("\n")
            if line.strip()
        ]
        last_message = json.loads(stdout_lines[-1])
        assert last_message["type"] == "complete"
        assert "insights_generated" in last_message
        assert "duration_secs" in last_message


class TestTrainerSubprocessExitCode:
    def test_exit_code_zero_on_success(
        self, training_database_path, tmp_path,
    ):
        models_path = str(tmp_path / "models")
        completed = subprocess.run(
            [
                sys.executable, "-m", "engram_trainer",
                "--database", training_database_path,
                "--models-path", models_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert completed.returncode == 0


class TestTrainerSubprocessMissingDatabase:
    def test_exit_code_nonzero_on_missing_database(self, tmp_path):
        models_path = str(tmp_path / "models")
        nonexistent_database = str(tmp_path / "nonexistent.db")
        completed = subprocess.run(
            [
                sys.executable, "-m", "engram_trainer",
                "--database", nonexistent_database,
                "--models-path", models_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert completed.returncode == 1

        stdout_lines = [
            line for line in completed.stdout.strip().split("\n")
            if line.strip()
        ]
        assert len(stdout_lines) >= 1
        error_message = json.loads(stdout_lines[-1])
        assert error_message["type"] == "error"
