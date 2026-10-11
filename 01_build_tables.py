import gzip
import json
import os
import tarfile
from multiprocessing import Pool

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data"


def read_day(path):
    calls = []
    tools = []
    tf = tarfile.open(path, "r:gz")
    for member in tf:
        if not member.name.endswith(".jsonl.gz"):
            continue
        f = gzip.open(tf.extractfile(member), "rt")
        for line in f:
            s = json.loads(line)
            sid = s["session_id"]
            n = 0
            for turn_no, t in enumerate(s["turns"]):
                for c in t["llm_calls"]:
                    tok = c["tokens"] or {}
                    # prompt split by role, token_len sums up to the prompt size
                    by_role = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
                    for m in c["message_metadata"] or []:
                        if m["role"] in by_role:
                            by_role[m["role"]] += m["token_len"] or 0
                    calls.append([sid, s["date"], turn_no, n, c["timestamp"], c["duration_ms"],
                                  c["initiator_type"], c["model"], c["result"], c["status_code"],
                                  tok.get("prompt"), tok.get("completion"), tok.get("cached"),
                                  by_role["system"], by_role["user"], by_role["assistant"], by_role["tool"]])
                    n += 1
                for b in t["tool_batches"]:
                    for fc in b["function_calls"]:
                        tools.append([sid, turn_no, b["timestamp"], fc["duration_ms"], b["auto_continue"],
                                      b["result"], fc["name"], fc["status"]])

    calls = pd.DataFrame(calls, columns=["session_id", "date", "turn", "call", "ts", "duration_ms",
                                         "initiator", "model", "result", "status_code",
                                         "prompt", "completion", "cached",
                                         "system_tokens", "user_tokens", "assistant_tokens", "tool_tokens"])
    tools = pd.DataFrame(tools, columns=["session_id", "turn", "ts", "duration_ms", "auto_continue",
                                         "batch_result", "name", "status"])
    calls["ts"] = pd.to_datetime(calls["ts"])
    tools["ts"] = pd.to_datetime(tools["ts"])

    day = os.path.basename(path).replace("date.", "").replace(".tar.gz", "")
    calls.to_parquet(f"{OUT_DIR}/calls/{day}.parquet", index=False)
    tools.to_parquet(f"{OUT_DIR}/tools/{day}.parquet", index=False)
    print(day, "sessions:", calls["session_id"].nunique(), "calls:", len(calls), "tool calls:", len(tools))


if __name__ == "__main__":
    os.makedirs(f"{OUT_DIR}/calls", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/tools", exist_ok=True)
    files = sorted(f"{RAW_DIR}/{f}" for f in os.listdir(RAW_DIR) if f.endswith(".tar.gz"))
    with Pool(4) as p:
        p.map(read_day, files)
