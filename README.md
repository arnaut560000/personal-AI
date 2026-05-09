# RoomAI

RoomAI is a local personal AI companion with memory, a custom animated interface, and a FastAPI backend.

## Setup

Install Ollama and pull the small local model:

```powershell
ollama pull qwen2.5:0.5b
```

Create the Python environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run RoomAI:

```powershell
.\.venv\Scripts\python.exe main.py
```

Open:

```text
http://localhost:8000
```

To open from another device on the same Wi-Fi:

```powershell
$env:ROOMAI_API_HOST="0.0.0.0"
$env:ROOMAI_API_PORT="8000"
.\.venv\Scripts\python.exe main.py
```

Then open:

```text
http://YOUR-PC-IP:8000
```

## Memory

RoomAI can remember facts through natural messages such as:

```text
my name is Arnaut
call me Arnaut
remember that my project is RoomAI
i am working on a personal AI companion
```
