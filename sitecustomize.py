# Auto-imported on interpreter startup if it's on sys.path (your CWD is).
import os, asyncio, sys

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        sys.stderr.write("[sitecustomize] Using WindowsProactorEventLoopPolicy\n")
    except Exception as e:
        sys.stderr.write(f"[sitecustomize] Failed to set policy: {e}\n")
