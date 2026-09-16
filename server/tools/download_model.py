"""Download one of the two built-in System One model profiles.

Run:
    uv run python -m tools.download_model default
    uv run python -m tools.download_model larger
"""

import argparse

from system_one_lite.engine import MODEL_PROFILES, resolve_model_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=MODEL_PROFILES)
    args = parser.parse_args()

    model_id = MODEL_PROFILES[args.profile]
    model_path, revision = resolve_model_snapshot(model_id)
    print(f"{args.profile}: {model_id}")
    print(f"revision: {revision or 'local'}")
    print(f"path: {model_path}")


if __name__ == "__main__":
    main()
