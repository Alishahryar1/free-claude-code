"""Deterministic JSONL peer for native transport tests; never executes tools."""

import json
import sys
import time
from pathlib import Path


def emit(value):
    encoded = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
    # Exercise framing across both large messages and UTF-8 boundaries.
    for start in range(0, len(encoded), 32767):
        sys.stdout.buffer.write(encoded[start : start + 32767])
        sys.stdout.buffer.flush()


mode = sys.argv[1]
defaults = {
    "approvalPolicy": "on-request",
    "approvalsReviewer": "user",
    "sandbox": {"type": "workspaceWrite"},
    "activePermissionProfile": {"id": ":workspace"},
}
creation_id = None
turn = "turn-1"
for line in sys.stdin.buffer:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        emit({"id": request_id, "result": {"userAgent": "codex/0.153.0"}})
    elif method == "initialized":
        pass
    elif method in {"thread/start", "thread/resume"}:
        if mode == "delayed-create":
            creation_id = request_id
        else:
            emit(
                {
                    "id": request_id,
                    "result": {"thread": {"id": "native-1", "turns": []}, **defaults},
                }
            )
    elif method == "test/barrier":
        emit({"id": request_id, "result": {}})
    elif method == "test/release-create":
        emit(
            {
                "id": creation_id,
                "result": {"thread": {"id": "native-1", "turns": []}, **defaults},
            }
        )
        emit({"id": request_id, "result": {}})
    elif method == "turn/start":
        if mode == "child-warning-on-close":
            turn = request["params"]["clientUserMessageId"]
        emit(
            {
                "method": "turn/started",
                "params": {
                    "threadId": "native-1",
                    "turn": {"id": turn, "status": "inProgress"},
                },
            }
        )
        if mode == "ordered-usage":
            emit(
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "native-1",
                        "turnId": turn,
                        "tokenUsage": {"last": {"totalTokens": 12_438}},
                    },
                }
            )
            emit(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "native-1",
                        "turn": {"id": turn, "status": "completed"},
                    },
                }
            )
        elif mode in {"prompt", "child-prompt"}:
            if mode == "child-prompt":
                emit(
                    {
                        "method": "thread/started",
                        "params": {
                            "thread": {
                                "id": "child-1",
                                "source": {
                                    "subAgent": {
                                        "thread_spawn": {
                                            "parent_thread_id": "native-1",
                                            "depth": 1,
                                        }
                                    }
                                },
                            }
                        },
                    }
                )
            emit(
                {
                    "method": "item/commandExecution/requestApproval",
                    "id": 0,
                    "params": {
                        "threadId": "child-1" if mode == "child-prompt" else "native-1",
                        "turnId": turn,
                        "itemId": "command",
                        "command": "echo test",
                        "availableDecisions": ["accept", "decline", "cancel"],
                    },
                }
            )
        else:
            emit(
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "native-1",
                        "turnId": turn,
                        "itemId": "answer",
                        "delta": "snow ☃ " * 16000,
                    },
                }
            )
            emit(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "native-1",
                        "turnId": turn,
                        "item": {
                            "id": "answer",
                            "type": "agentMessage",
                            "text": "snow ☃ " * 16000,
                        },
                    },
                }
            )
            emit(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "native-1",
                        "turn": {"id": turn, "status": "completed"},
                    },
                }
            )
        emit(
            {"id": request_id, "result": {"turn": {"id": turn, "status": "inProgress"}}}
        )
    elif method is None and request_id == 0:
        emit(
            {
                "method": "serverRequest/resolved",
                "params": {"threadId": "native-1", "requestId": 0},
            }
        )
        emit(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "native-1",
                    "turnId": turn,
                    "item": {
                        "id": "answer",
                        "type": "agentMessage",
                        "text": request["result"]["decision"],
                    },
                },
            }
        )
        emit(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "native-1",
                    "turn": {"id": turn, "status": "completed"},
                },
            }
        )
    elif method == "test/eof":
        break
    elif method == "test/malformed":
        sys.stdout.buffer.write(b"invalid-json\n")
        sys.stdout.buffer.flush()
    elif method == "test/malformed-flood":
        sys.stdout.buffer.write(b"invalid-json\n" + b"x" * 500_000 + b"\n")
        sys.stdout.buffer.flush()
    else:
        emit({"id": request_id, "result": {}})

if mode == "child-warning-on-close":
    emit(
        {
            "method": "guardianWarning",
            "params": {"threadId": "child", "message": "Child is shutting down"},
        }
    )
    # Keep stdout alive until the parent has processed this close-time event.
    while not Path(sys.argv[2]).exists():
        time.sleep(0.01)
