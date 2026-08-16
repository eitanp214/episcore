"""Probe which Egocentric-10K mirrors are reachable without gated access."""

import os

from huggingface_hub import HfApi, get_token

CANDIDATES = [
    "builddotai/Egocentric-10K",
    "Voxel51/Egocentric_10K_subset",
    "Voxel51/Egocentric_10K_Evaluation",
    "builddotai/Egocentric-10K-Evaluation",
]


def main() -> None:
    env_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"HF token in env: {env_token}")
    try:
        print(f"HF token stored: {bool(get_token())}")
    except Exception as exc:  # noqa: BLE001
        print(f"stored-token check failed: {exc}")

    api = HfApi()
    for repo in CANDIDATES:
        print(f"\n=== {repo}")
        try:
            info = api.dataset_info(repo)
            print(f"  gated={info.gated}  private={info.private}")
            names = [s.rfilename for s in (info.siblings or [])]
            print(f"  n_files={len(names)}")
            for n in names[:10]:
                print(f"    {n}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
