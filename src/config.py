from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


def load_config(path=None):
    """Load project/workspace config.

    V5 adds HGAC_CONFIG_PATH so the regional app can run many independent bayou
    workspaces without rewriting the root project's config.yaml.
    """
    load_dotenv()
    if path is None:
        path = os.getenv("HGAC_CONFIG_PATH", "config/config.yaml")
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = path.parent.parent
    cfg["_root"] = str(root)
    cfg["_config_path"] = str(path)

    for key, value in list(cfg.get("paths", {}).items()):
        if value in (None, ""):
            continue
        p = Path(str(value))
        cfg["paths"][key] = str(p if p.is_absolute() else (root / p).resolve())

    aoi = cfg.get("aoi", {})
    if aoi.get("path"):
        p = Path(str(aoi["path"]))
        aoi["path"] = str(p if p.is_absolute() else (root / p).resolve())

    cfg.setdefault("secrets", {})
    cfg["secrets"]["floodhub_api_key"] = os.getenv("FLOODHUB_API_KEY", "").strip()
    env_gid = os.getenv("FLOODHUB_GAUGE_ID", "").strip()
    cfg.setdefault("floodhub", {})
    if env_gid:
        cfg["floodhub"]["gauge_id"] = env_gid
    return cfg


def ensure_directories(cfg):
    for key in ("raw", "processed", "outputs"):
        value = cfg.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
