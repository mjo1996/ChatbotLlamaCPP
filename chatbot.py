#!/usr/bin/env python3
"""Simple chatbot using a local GGUF model via llama.cpp.

Features:
- Loads a local .gguf model with llama-cpp-python (CPU by default).
- User-configurable context length and temperature.
- Replies are streamed to the terminal token by token as they are generated.
- Reasoning ("thinking") is stripped from the stream, so only the answer shows.
- Conversation history is fed back to the model for context.
- Chats are persisted to disk as JSON files (can be disabled with /nohistory).
- A new chat is opened by default; user can start a new chat at any time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

try:
    from llama_cpp import Llama
    from llama_cpp import llama_chat_format
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


# The tags a reasoning model wraps its thinking in. Spelled in pieces so tools
# that treat angle-bracket text as markup cannot mangle them in this file.
THINK_OPEN_TAG = "<" + "think" + ">"
THINK_CLOSE_TAG = "</" + "think" + ">"
# Only announce thinking once a reasonable amount of it has been hidden, so an
# empty thinking block (the shape Qwen3 emits when thinking is off) never
# flickers a status message on screen.
THINK_NOTICE_CHARS = 40


class ThinkingStripper:
    """Strip a leading reasoning block from a token stream.

    Reasoning models wrap their reasoning in ``think`` tags that the chat
    template opens *before* the answer, so the reasoning arrives interleaved in
    the same text stream as the visible reply. This filter drops everything
    between the tags and passes the rest through as soon as it is unambiguous,
    which keeps streaming live instead of buffering the whole reply.

    ``starts_inside`` covers the other half of the story: some templates (Qwen3
    among them) open the reasoning block in the *prompt*, so the stream begins
    mid-reasoning with no opening tag to spot and only the closing tag to end
    it. See :func:`prompt_opens_thinking`.
    """

    def __init__(
        self,
        open_tag: str = THINK_OPEN_TAG,
        close_tag: str = THINK_CLOSE_TAG,
        starts_inside: bool = False,
    ) -> None:
        self._open = open_tag
        self._close = close_tag
        self._buffer = ""
        self._decided = starts_inside
        self.in_thinking = starts_inside
        # Swallow the newlines a template puts between the reasoning and the
        # answer, even when a chunk boundary lands right after the closing tag.
        self._skip_space = starts_inside
        self.hidden_chars = 0

    def feed(self, text: str) -> str:
        """Consume newly streamed ``text`` and return the visible part of it."""
        if not text:
            return ""
        self._buffer += text
        visible: list[str] = []
        while self._buffer:
            if self.in_thinking:
                end = self._buffer.find(self._close)
                if end == -1:
                    # Keep a tail: the closing tag may straddle two chunks.
                    keep = len(self._close) - 1
                    self.hidden_chars += max(0, len(self._buffer) - keep)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self.hidden_chars += end + len(self._close)
                self._buffer = self._buffer[end + len(self._close):].lstrip("\n")
                self.in_thinking = False
                self._skip_space = True
                continue
            if not self._decided:
                stripped = self._buffer.lstrip()
                if stripped.startswith(self._open):
                    self._buffer = stripped[len(self._open):]
                    self.in_thinking = True
                    continue
                if self._open.startswith(stripped):
                    # Might still become the opening tag; wait for more tokens.
                    break
                self._decided = True
                self._buffer = stripped
            elif self._skip_space:
                stripped = self._buffer.lstrip()
                if not stripped:
                    # Whitespace only so far: hold it back and wait.
                    self._buffer = ""
                    break
                self._skip_space = False
                self._buffer = stripped
            visible.append(self._buffer)
            self._buffer = ""
        return "".join(visible)

    def flush(self) -> str:
        """Return whatever is left to print once the stream has ended."""
        if self.in_thinking:
            # An unterminated thinking block is all reasoning; drop it.
            self._buffer = ""
            return ""
        tail, self._buffer = self._buffer, ""
        return tail


def prompt_opens_thinking(
    llm: Llama, enabled: bool | None, messages: list[dict] | None = None
) -> bool:
    """Report whether the chat template leaves the prompt inside a think block.

    Renders the model's own chat template, so the answer reflects what the model
    is actually prompted with rather than a guess about where tags live. When
    this is true, everything the model generates is reasoning until it emits the
    closing tag, and the reasoning has no opening tag of its own to detect.
    """
    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        return False
    try:
        formatter = llama_chat_format.Jinja2ChatFormatter(
            template=template,
            eos_token="",
            bos_token="",
            stop_token_ids=[],
        )
        extra = {} if enabled is None else {"enable_thinking": enabled}
        probe = messages or [{"role": "user", "content": "hi"}]
        rendered = formatter(messages=probe, **extra).prompt
    except Exception:
        return False
    return rendered.rstrip().endswith(THINK_OPEN_TAG)


ASSISTANT_PROMPT = "Assistant: "


def show_thinking_hint(active: bool) -> None:
    """Signal that the model is reasoning, without printing the reasoning.

    Written in place so the hint can be erased again as soon as the answer
    starts. Only used on a real terminal: when output is redirected or piped,
    the hint would just be noise in the captured text.
    """
    if not sys.stdout.isatty():
        return
    if active:
        sys.stdout.write("(thinking...)")
    else:
        # Erase the hint and restore the prompt prefix.
        sys.stdout.write(f"\r\033[K{ASSISTANT_PROMPT}")
    sys.stdout.flush()


def _iter_stream_text(stream: Iterator[dict], chat_completion: bool) -> Iterator[str]:
    """Yield the text of each chunk of a llama.cpp completion stream."""
    for chunk in stream:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if chat_completion:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                # Some builds expose reasoning out-of-band; it is never shown.
                continue
            piece = delta.get("content") or ""
        else:
            piece = choice.get("text") or ""
        if piece:
            yield piece


def generate_reply(
    llm: Llama,
    messages: list[dict],
    temperature: float,
    on_thinking: Callable[[bool], None] | None = None,
    thinking: dict | None = None,
) -> Iterator[str]:
    """Stream the assistant's reply, with any reasoning stripped out.

    Yields visible text chunks as they arrive. ``thinking`` is the state dict
    from :func:`install_thinking_control`, used to tell whether the model was
    prompted mid-reasoning. ``on_thinking`` is called with ``True``/``False`` as
    a suppressed reasoning block starts and ends, so the caller can show a
    status hint while the model thinks.
    """
    starts_inside = False
    try:
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        chat_completion = True
    except Exception:
        prompt = build_prompt(messages)
        stream = llm(
            prompt,
            temperature=temperature,
            max_tokens=1024,
            stop=["<|im_end|>", "<|im_start|>"],
            stream=True,
        )
        chat_completion = False

    if chat_completion and thinking is not None:
        # The template may have opened the reasoning block in the prompt itself,
        # in which case the whole run up to the closing tag is reasoning.
        starts_inside = prompt_opens_thinking(llm, thinking["enabled"], messages)

    stripper = ThinkingStripper(starts_inside=starts_inside)
    announced = False
    for piece in _iter_stream_text(stream, chat_completion):
        visible = stripper.feed(piece)
        if on_thinking is not None:
            if (
                stripper.in_thinking
                and not announced
                and stripper.hidden_chars >= THINK_NOTICE_CHARS
            ):
                announced = True
                on_thinking(True)
            elif announced and not stripper.in_thinking:
                announced = False
                on_thinking(False)
        if visible:
            yield visible
    if announced and on_thinking is not None:
        on_thinking(False)
    tail = stripper.flush()
    if tail:
        yield tail


def str_to_bool(value: str) -> bool:
    """Parse a human-friendly boolean such as ``true``/``false`` or ``on``/``off``."""
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on", "y"):
        return True
    if normalized in ("0", "false", "no", "off", "n"):
        return False
    raise ValueError(f"expected 'true' or 'false', got {value!r}")


def argparse_bool(value: str) -> bool:
    """argparse ``type=`` wrapper that reports bad booleans nicely."""
    try:
        return str_to_bool(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def install_thinking_control(llm: Llama) -> dict:
    """Let the chat template's ``enable_thinking`` variable be toggled at runtime.

    ``enable_thinking`` is the llama.cpp-compatible reasoning switch: chat
    templates that support it (e.g. Qwen3) emit an empty reasoning block when it
    is ``False`` and think normally when it is ``True`` or absent. Templates that
    do not reference the variable simply ignore the extra argument.

    Returns a state dict whose ``"enabled"`` key holds ``True`` (think), ``False``
    (do not think) or ``None`` (leave the model's own default untouched).
    """
    base_handler = (
        llm.chat_handler
        or llm._chat_handlers.get(llm.chat_format)
        or llama_chat_format.get_chat_completion_handler(llm.chat_format)
    )
    state: dict = {"enabled": None}

    def handler_with_thinking(*args, **kwargs):
        if state["enabled"] is not None:
            kwargs["enable_thinking"] = state["enabled"]
        return base_handler(*args, **kwargs)

    llm.chat_handler = handler_with_thinking
    return state


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /new            Start a new chat\n"
        "  /list           List saved chats\n"
        "  /load <id>      Load a saved chat by id\n"
        "  /history        Show messages in current chat\n"
        "  /save           Force-save the current chat\n"
        "  /nohistory      Toggle saving chats to disk (on/off)\n"
        "  /temp <value>   Change temperature for subsequent replies\n"
        "  /think <value>  Enable/disable thinking (true/false), or show current state\n"
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
        "--think",
        type=argparse_bool,
        default=None,
        metavar="true|false",
        help=(
            "Enable or disable 'thinking' (reasoning) for models that offer both "
            "modes, e.g. --think=false. Sets the chat template's enable_thinking "
            "variable (llama.cpp-compatible); models whose template does not "
            "define it are unaffected. Default: keep the model's own default."
        ),
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
    p.add_argument(
        "--no-history",
        action="store_true",
        help=(
            "Start with history saving disabled: nothing is written to the chat "
            "folder for the whole session. Toggle it at runtime with /nohistory."
        ),
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
    think_label = (
        "model default" if args.think is None else ("on" if args.think else "off")
    )
    print(
        f"  n_ctx={n_ctx}  temperature={temperature}  "
        f"gpu_layers={args.gpu_layers}  thinking={think_label}"
    )

    llm_kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": args.gpu_layers,  # 0 = CPU only (default)
        "verbose": False,
    }
    if args.threads is not None:
        llm_kwargs["n_threads"] = args.threads

    llm = Llama(**llm_kwargs)

    thinking = install_thinking_control(llm)
    thinking["enabled"] = args.think

    save_enabled = not args.no_history
    chat = new_chat(args.system)
    if save_enabled:
        ensure_history_dir()
        save_chat(chat)
    print(f"\nStarted new chat: {chat['id']}")
    if not save_enabled:
        print("History saving is disabled (--no-history): nothing will be saved.")
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
                if save_enabled:
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
                if save_enabled:
                    save_chat(chat)
                    print(f"Saved to {chat_path(chat['id'])}\n")
                else:
                    print(
                        "History saving is disabled (/nohistory); nothing written. "
                        "Enable it first to save.\n"
                    )
            elif cmd == "/nohistory":
                if not arg:
                    save_enabled = not save_enabled
                else:
                    try:
                        save_enabled = str_to_bool(arg)
                    except ValueError as exc:
                        print(f"{exc}\n")
                        continue
                if save_enabled:
                    print("History saving enabled: chats will be written to disk.\n")
                else:
                    print(
                        "History saving disabled: nothing will be written to the "
                        "chat folder.\n"
                    )
            elif cmd == "/temp":
                if not arg:
                    print(f"Current temperature: {temperature}\n")
                    continue
                try:
                    temperature = float(arg)
                    print(f"Temperature set to {temperature}\n")
                except ValueError:
                    print("Invalid temperature value.\n")
            elif cmd == "/think":
                if not arg:
                    current = thinking["enabled"]
                    shown = (
                        "model default"
                        if current is None
                        else ("on" if current else "off")
                    )
                    print(f"Current thinking: {shown}\n")
                    continue
                try:
                    thinking["enabled"] = str_to_bool(arg)
                    print(
                        f"Thinking {'enabled' if thinking['enabled'] else 'disabled'}\n"
                    )
                except ValueError as exc:
                    print(f"{exc}\n")
            else:
                print(f"Unknown command: {cmd}. Type /help for help.\n")
            continue

        chat["messages"].append({"role": "user", "content": user_input})

        sys.stdout.write(f"\n{ASSISTANT_PROMPT}")
        sys.stdout.flush()
        parts: list[str] = []
        try:
            for piece in generate_reply(
                llm,
                chat["messages"],
                temperature,
                on_thinking=show_thinking_hint,
                thinking=thinking,
            ):
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n[generation interrupted]")
        except Exception as e:
            print(f"\n[error generating reply: {e}]")
            chat["messages"].pop()
            continue

        reply = "".join(parts).strip()
        if not reply:
            # Nothing visible came back: the model was still thinking when it
            # stopped (context or token limit), or it produced no output at all.
            chat["messages"].pop()
            print("[no reply produced - the model stopped before answering]\n")
            continue

        chat["messages"].append({"role": "assistant", "content": reply})
        if save_enabled:
            save_chat(chat)
        print("\n")

    if save_enabled:
        save_chat(chat)
        print(f"Chat saved: {chat_path(chat['id'])}")
    else:
        print("History saving disabled; chat not saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
