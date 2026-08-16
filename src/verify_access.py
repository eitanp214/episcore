"""Confirm the token is readable and the gated repo actually opens.

Checks three things independently, because they fail for different reasons:
a token can exist without the dataset terms being accepted, and terms can be
accepted in a browser session the CLI never saw.
"""

from __future__ import annotations

import sys

from huggingface_hub import HfApi, HfFileSystem, get_token, whoami

REPO = "builddotai/Egocentric-10K"


def main() -> int:
    token = get_token()
    print(f"1. token present: {bool(token)}")
    if not token:
        print("   -> run: D:\\episcore\\.venv\\Scripts\\huggingface-cli.exe login")
        return 1

    try:
        me = whoami()
        print(f"2. authenticated as: {me.get('name')} ({me.get('email', 'no email')})")
    except Exception as exc:  # noqa: BLE001
        print(f"2. auth FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        print("   -> token is present but rejected; create a fresh Read token")
        return 1

    api = HfApi()
    try:
        info = api.dataset_info(REPO)
        print(f"3. repo metadata: gated={info.gated}")
    except Exception as exc:  # noqa: BLE001
        print(f"3. metadata FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    # The decisive test: can we actually pull bytes from a gated shard?
    fs = HfFileSystem()
    probe = (
        f"datasets/{REPO}/factory_001/workers/worker_001/"
        "factory001_worker001_part00.tar"
    )
    try:
        with fs.open(probe, "rb") as handle:
            head = handle.read(4096)
        print(f"4. gated shard readable: YES ({len(head)} bytes pulled)")
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        print(f"4. gated shard readable: NO -- {name}: {str(exc)[:200]}")
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            print(
                "   -> the token works but access was never granted.\n"
                f"      Open https://huggingface.co/datasets/{REPO} and click\n"
                "      'Agree and access repository', then rerun this."
            )
        return 1

    print("\nACCESS CONFIRMED — cross-factory run can start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
