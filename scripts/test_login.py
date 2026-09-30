#!/usr/bin/env python3
"""Verify Choice login WITHOUT the GTK activator — uses username/password.

    # put creds in config.yaml (choice.username/password), OR:
    EM_USERNAME=you EM_PASSWORD=secret python3 scripts/test_login.py

Prints a clear PASS/FAIL and, on success, pulls one live fundamental to prove
data access works end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.config import Config
from quant.choice_client import ChoiceClient


def main() -> None:
    cfg = Config.load()
    user = cfg.get("choice.username", "")
    if not user:
        print("✗ No credentials found.")
        print("  Set choice.username / choice.password in config.yaml,")
        print("  or export EM_USERNAME=... EM_PASSWORD=...  then re-run.")
        return

    print(f"Attempting Choice login as '{user}' (headless, no activator) ...")
    client = ChoiceClient(username=user,
                          password=cfg.get("choice.password", ""),
                          start_options=cfg.get("choice.start_options", "ForceLogin=1"))
    ok = client.login()
    if not ok:
        print("✗ Login FAILED (see warning above).")
        print("  Common causes: wrong creds, account not entitled to the Python")
        print("  quant API, or this account type requires one-time activation.")
        return

    print("✓ Login PASSED — Choice is LIVE.")
    try:
        df = client.css("300750.SZ", "SECURITYNAME,PETTM,MV", TradeDate="")
        print("  sample fundamental fetch:")
        print(df.to_string())
    except Exception as e:
        print(f"  (login ok, sample fetch errored: {e})")
    client.close()


if __name__ == "__main__":
    main()
