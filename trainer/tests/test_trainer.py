import json


class TestRunMediumOutputsJsonLines:
    def test_produces_json_lines_with_progress_and_complete(
        self, test_database_path, tmp_path, capsys,
    ):
        from engram_trainer.trainer import run_medium

        models_path = str(tmp_path / "models")
        run_medium(test_database_path, models_path)

        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert len(lines) >= 2

        parsed_messages = [json.loads(line) for line in lines]
        message_types = [msg["type"] for msg in parsed_messages]

        assert "progress" in message_types
        assert "complete" in message_types

        complete_message = next(
            msg for msg in parsed_messages if msg["type"] == "complete"
        )
        assert "insights_generated" in complete_message
        assert "duration_secs" in complete_message


class TestRunMediumEmptyDatabase:
    def test_no_crash_emits_complete_with_zero_insights(
        self, empty_database_path, tmp_path, capsys,
    ):
        from engram_trainer.trainer import run_medium

        models_path = str(tmp_path / "models")
        run_medium(empty_database_path, models_path)

        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        parsed_messages = [json.loads(line) for line in lines]

        complete_message = next(
            msg for msg in parsed_messages if msg["type"] == "complete"
        )
        assert complete_message["insights_generated"] == 0


class TestRunDeepCallsMedium:
    def test_produces_same_output_structure(
        self, test_database_path, tmp_path, capsys,
    ):
        from engram_trainer.trainer import run_deep

        models_path = str(tmp_path / "models")
        run_deep(test_database_path, models_path)

        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        parsed_messages = [json.loads(line) for line in lines]
        message_types = [msg["type"] for msg in parsed_messages]

        assert "progress" in message_types
        assert "complete" in message_types
