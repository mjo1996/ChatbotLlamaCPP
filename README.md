# ChatbotLlamaCPP

A simple Python CLI chatbot that loads a local **GGUF** model via
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

## Features

- Loads any local `.gguf` model.
- **Replies stream to the terminal** as they are generated, token by token.
- **Reasoning is never printed** — when the model thinks, the reasoning block is
  filtered out of the stream and only the final answer is shown.
- User-configurable **context length** (`n_ctx`) and **temperature**.
- **CPU by default** (`--gpu-layers 0`); pass `--gpu-layers N` to offload.
- **Thinking/reasoning can be toggled** (`--think true|false`) for models that
  support both modes (e.g. Qwen3), using the llama.cpp-compatible
  `enable_thinking` chat-template variable.
- Conversation **history is sent back to the model** so it remembers context.
- Chats are **persisted to disk** as JSON in `~/.chatbot_llamacpp/chats/`.
- A **new chat is opened by default**; create more chats anytime with `/new`.
- Saving can be **disabled at runtime** with `/nohistory`, or from the start
  with `--no-history`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python chatbot.py --model /path/to/model.gguf
```

You will be prompted for context length and temperature (or pass `--ctx` / `--temp`).

```bash
python chatbot.py -m model.gguf --ctx 4096 --temp 0.7
```

### Disabling history

By default chats are saved to `~/.chatbot_llamacpp/chats/`. To keep a session
entirely off disk, start with `--no-history`:

```bash
python chatbot.py -m model.gguf --no-history
```

Nothing is written for the whole session (no new chat file is created). You can
also toggle this at any time during a session with the `/nohistory` command.

### Thinking / reasoning

For models that can both "think" and answer directly (Qwen3-style reasoning
models), control it with `--think`:

```bash
python chatbot.py -m model.gguf --think=false   # answer directly, no reasoning
python chatbot.py -m model.gguf --think=true    # allow the model to reason first
```

`--think` sets the chat template's `enable_thinking` variable, which is the
same switch llama.cpp's server exposes (`--chat-template-kwargs
'{"enable_thinking": false}'`). Models whose chat template does not define that
variable ignore it. If `--think` is omitted, the model's own default is used.
You can also toggle it mid-session with `/think`.

Either way the reasoning itself is **not printed**: while the model thinks, the
reply line shows a `(thinking...)` hint (on a terminal only), and the answer
starts streaming as soon as the model finishes thinking. Only the final answer —
with no reasoning and no tag markers — is stored in the chat history.

## In-chat commands

```
/new            Start a new chat
/list           List saved chats
/load <id>      Load a saved chat by id
/history        Show messages in current chat
/save           Force-save the current chat
/nohistory      Toggle saving chats to disk (on/off)
/temp <value>   Change temperature for subsequent replies
/think <value>  Enable/disable thinking (true/false), or show current state
/help           Show help
/exit, /quit    Exit
```

Chats are saved automatically after each assistant reply. Use `/nohistory` to
disable all saving to disk for the session — while disabled, no chat file is
created or updated (existing saved chats are left untouched), and `/save` will
refuse to write until saving is re-enabled. Run `/nohistory` again (or
`/nohistory on`) to turn saving back on.
