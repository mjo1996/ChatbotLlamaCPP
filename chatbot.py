#!/usr/bin/env python3
"""Simple chatbot using a local GGUF model via llama.cpp.

Features:
- Loads a local .gguf model with llama-cpp-python (CPU by default).
- User-configurable context length and temperature.
- Conversation history is fed back to the model for context.
- Chats are persisted to disk as JSON files.
- A new chat is opened by default; user can start a new chat at any time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    sys.stderr.write(
        "Error: llama-cpp-python is not installed.\n"
        "Install it with:  pip install llama-cpp-python\n"
    )
    sys.exit(1)


HISTORY_DIR = Path.home() / ".chatbot_llamacpp" / "chats"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def ensure_history_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def new_chat(system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict:
    chat_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    return {
        "id": chat_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [{"role": "system", "content": system_prompt}],
    }


def chat_path(chat_id: str) -> Path:
    return HISTORY_DIR / f"{chat_id}.json"


def save_chat(chat: dict) -> None:
    ensure_history_dir()
    with chat_path(chat["id"]).open("w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2, ensure_ascii=False)


def load_chat(chat_id: str) -> dict | None:
    p = chat_path(chat_id)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_chats() -> list[dict]:
    ensure_history_dir()
    chats = []
    for p in sorted(HISTORY_DIR.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            user_msgs = [m for m in data.get("messages", []) if m["role"] == "user"]
            preview = user_msgs[0]["content"][:60] if user_msgs else "(empty)"
            chats.append(
                {
                    "id": data.get("id", p.stem),
                    "created_at": data.get("created_at", ""),
                    "preview": preview,
                    "n_messages": len(data.get("messages", [])),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return chats


def build_prompt(messages: list[dict]) -> str:
    """Build a simple ChatML-ish prompt that works with most instruct models."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def generate_reply(llm: Llama, messages: list[dict], temperature: float) -> str:
    """Use llama.cpp's chat completion if available, else raw completion."""
    try:
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        prompt = build_prompt(messages)
        resp = llm(
            prompt,
            temperature=temperature,
            max_tokens=1024,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        return resp["choices"][0]["text"].strip()


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /new            Start a new chat\n"
        "  /list           List saved chats\n"
        "  /load <id>      Load a saved chat by id\n"
        "  /history        Show messages in current chat\n"
        "  /save           Force-save the current chat\n"
        "  /temp <value>   Change temperature for subsequent replies\n"
        "  /help           Show this help\n"
        "  /exit, /quit    Exit the program\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple llama.cpp chatbot")
    p.add_argument("--model", "-m", required=True, help="Path to local .gguf model file")
    p.add_argument(
        "--ctx",
        "-c",
        type=int,
        default=None,
        help="Context length (n_ctx). If omitted, you will be prompted.",
    )
    p.add_argument(
        "--temp",
        "-t",
        type=float,
        default=None,
        help="Sampling temperature. If omitted, you will be prompted.",
    )
    p.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU. Default 0 (CPU only).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: llama.cpp auto).",
    )
    p.add_argument(
        "--system",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for new chats.",
    )
    return p.parse_args()


def prompt_float(msg: str, default: float) -> float:
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def prompt_int(msg: str, default: int) -> int:
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def main() -> int:
    args = parse_args()

    model_path = os.path.expanduser(args.model)
    if not os.path.isfile(model_path):
        sys.stderr.write(f"Error: model file not found: {model_path}\n")
        return 1

    n_ctx = args.ctx if args.ctx is not None else prompt_int("Context length", 8096)
    temperature = (
        args.temp if args.temp is not None else prompt_float("Temperature", 0.7)
    )

    print(f"\nLoading model: {model_path}")
    print(f"  n_ctx={n_ctx}  temperature={temperature}  gpu_layers={args.gpu_layers}")

    llm_kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": args.gpu_layers,  # 0 = CPU only (default)
        "verbose": False,
    }
    if args.threads is not None:
        llm_kwargs["n_threads"] = args.threads

    llm = Llama(**llm_kwargs)

    ensure_history_dir()
    chat = new_chat(args.system)
    save_chat(chat)
    print(f"\nStarted new chat: {chat['id']}")
    print("Type your message, or /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, *rest = user_input.split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/new":
                chat = new_chat(args.system)
                save_chat(chat)
                print(f"Started new chat: {chat['id']}\n")
            elif cmd == "/list":
                chats = list_chats()
                if not chats:
                    print("No saved chats.\n")
                else:
                    for c in chats:
                        print(
                            f"  {c['id']}  ({c['n_messages']} msgs)  {c['created_at']}  | {c['preview']}"
                        )
                    print()
            elif cmd == "/load":
                if not arg:
                    print("Usage: /load <chat-id>\n")
                    continue
                loaded = load_chat(arg.strip())
                if loaded is None:
                    print(f"No chat found with id: {arg}\n")
                else:
                    chat = loaded
                    print(
                        f"Loaded chat {chat['id']} ({len(chat['messages'])} messages)\n"
                    )
            elif cmd == "/history":
                for m in chat["messages"]:
                    print(f"[{m['role']}] {m['content']}")
                print()
            elif cmd == "/save":
                save_chat(chat)
                print(f"Saved to {chat_path(chat['id'])}\n")
            elif cmd == "/temp":
                if not arg:
                    print(f"Current temperature: {temperature}\n")
                    continue
                try:
                    temperature = float(arg)
                    print(f"Temperature set to {temperature}\n")
                except ValueError:
                    print("Invalid temperature value.\n")
            else:
                print(f"Unknown command: {cmd}. Type /help for help.\n")
            continue

        chat["messages"].append({"role": "user", "content": user_input})

        try:
            reply = generate_reply(llm, chat["messages"], temperature)
        except Exception as e:
            print(f"[error generating reply: {e}]\n")
            chat["messages"].pop()
            continue

        chat["messages"].append({"role": "assistant", "content": reply})
        save_chat(chat)
        print(f"\nAssistant: {reply}\n")

    save_chat(chat)
    print(f"Chat saved: {chat_path(chat['id'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
