"""Thin client for the CollegeFootballData.com (CFBD) REST API.

- Reads the key from the CFBD_API_KEY environment variable (never hard-code it).
- Caches every response under data/raw/ (git-ignored) so re-runs don't spend API calls.
- Counts calls made in this process so jobs can report quota use.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = os.environ.get("CFBD_BASE_URL", "https://api.collegefootballdata.com")
CACHE_DIR = Path(os.environ.get("CFBD_CACHE_DIR", "data/raw"))


class CFBDError(RuntimeError):
    pass


class CFBDClient:
    def __init__(self, api_key: str | None = None, use_cache: bool = True):
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise CFBDError("CFBD_API_KEY is not set")
        self.use_cache = use_cache
        self.calls = 0
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        )

    def _cache_path(self, path: str, params: dict) -> Path:
        key = path.strip("/") + "?" + json.dumps(params, sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return CACHE_DIR / path.strip("/").replace("/", "_") / f"{digest}.json"

    def get(self, path: str, refresh: bool = False, **params):
        params = {k: v for k, v in params.items() if v is not None}
        cache_file = self._cache_path(path, params)
        if self.use_cache and not refresh and cache_file.exists():
            return json.loads(cache_file.read_text())

        url = f"{BASE_URL}/{path.lstrip('/')}"
        for attempt in range(3):
            resp = self.session.get(url, params=params, timeout=60)
            self.calls += 1
            if resp.status_code == 429 and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            # Never echo the request headers (they hold the key)
            raise CFBDError(f"{path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if self.use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
        return data
