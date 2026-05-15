# ChatbotLlamaCPP

A simple Python CLI chatbot that loads a local **GGUF** model via
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

## Features

- Loads any local `.gguf` model.
- User-configurable **context length** (`n_ctx`) and **temperature**.
- **CPU by default** (`--gpu-layers 0`); pass `--gpu-layers N` to offload.
- Conversation **history is sent back to the model** so it remembers context.
- Chats are **persisted to disk** as JSON in `~/.chatbot_llamacpp/chats/`.
- A **new chat is opened by default**; create more chats anytime with `/new`.

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

## In-chat commands

```
/new            Start a new chat
/list           List saved chats
/load <id>      Load a saved chat by id
/history        Show messages in current chat
/save           Force-save the current chat
/temp <value>   Change temperature for subsequent replies
/help           Show help
/exit, /quit    Exit
```

Chats are saved automatically after each assistant reply.
