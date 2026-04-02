import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        description="Engram trainer: analyze memories and generate insights",
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Path to engram SQLite database",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="Directory for trained model artifacts",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        default=False,
        help="Run deep analysis (longer, more thorough)",
    )

    arguments = parser.parse_args()

    try:
        from engram_trainer.trainer import run_medium, run_deep

        if arguments.deep:
            run_deep(arguments.database, arguments.models_path)
        else:
            run_medium(arguments.database, arguments.models_path)
    except ImportError:
        error_message = json.dumps(
            {"type": "error", "message": "trainer module not yet implemented"},
        )
        print(error_message, flush=True)
        sys.exit(1)
    except Exception as exception:
        error_message = json.dumps(
            {"type": "error", "message": str(exception)},
        )
        print(error_message, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
