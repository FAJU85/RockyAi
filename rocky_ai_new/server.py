#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import sqlite3
import urllib.request
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), "rocky.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_system(level: str, message: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO system_logs(level, message, created_at) VALUES (?, ?, ?)",
        (level, message, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        if not self.is_configured():
            return (
                "DeepSeek API key missing. Set DEEPSEEK_API_KEY and restart."
            )
        try:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(
                    {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are Rocky, a professional biostatistics agent. "
                                    "You can perform statistical analysis, generate R code, "
                                    "suggest charts, and produce Jupyter snippets."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                ).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
        except Exception as e:  # keep robust
            log_system("ERROR", f"DeepSeek error: {e}")
            return f"DeepSeek error: {e}"


class RockyHandler(http.server.BaseHTTPRequestHandler):
    client = DeepSeekClient()

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "deepseek_configured": self.client.is_configured(),
                }
            )
            return

        if self.path == "/":
            self._send_html(self._index_html())
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send_json({"error": "Not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        user_message = (data.get("message") or "").strip()
        if not user_message:
            self._send_json({"error": "Empty message"}, 400)
            return

        reply = self.client.chat(user_message)

        # persist
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations(user_message, ai_response, created_at) VALUES (?, ?, ?)",
            (user_message, reply, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()

        self._send_json({"response": reply})

    def _index_html(self) -> str:
        return (
            """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Rocky AI - DeepSeek</title>
  <style>
    body{font-family:Arial;margin:0;padding:20px;background:#f5f5f5}
    .container{max-width:860px;margin:0 auto;background:#fff;border-radius:10px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,.1)}
    .chat{border:1px solid #ddd;border-radius:10px;height:420px;overflow:auto;padding:15px;background:#fafafa;margin-bottom:12px}
    .msg{margin:10px 0;padding:10px;border-radius:8px}
    .user{background:#3498db;color:#fff;margin-left:20%}
    .ai{background:#ecf0f1;color:#2c3e50;margin-right:20%}
    .row{display:flex;gap:8px}
    input{flex:1;padding:12px;border:1px solid #ddd;border-radius:6px}
    button{padding:12px 16px;border:0;background:#27ae60;color:#fff;border-radius:6px;cursor:pointer}
    .buttons{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
    .btn{padding:6px 12px;background:#95a5a6;color:#fff;border:none;border-radius:6px;cursor:pointer}
  </style>
  <script>
    async function sendMessage(msg){
      const input = document.getElementById('in');
      const text = (msg || input.value || '').trim();
      if(!text) return;
      add(text,'user');
      input.value='';
      const typing = add('Thinking...','ai');
      try{
        const res = await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
        const data = await res.json();
        typing.remove();
        add(data.response || 'No response','ai');
      }catch(e){ typing.remove(); add('Error: '+e,'ai'); }
    }
    function add(text,who){
      const c = document.getElementById('chat');
      const d = document.createElement('div');
      d.className = 'msg ' + (who==='user'?'user':'ai');
      d.innerHTML = '<b>'+(who==='user'?'You':'Rocky')+':</b> '+text.replace(/\n/g,'<br>');
      c.appendChild(d); c.scrollTop = c.scrollHeight; return d;
    }
  </script>
  </head>
<body>
  <div class="container">
    <h2>🤖 Rocky AI (DeepSeek)</h2>
    <div class="buttons">
      <button class="btn" onclick="sendMessage('Generate R code for t-test between two groups')">R code</button>
      <button class="btn" onclick="sendMessage('Suggest a chart for correlation analysis and provide code')">Chart</button>
      <button class="btn" onclick="sendMessage('Create a Jupyter notebook snippet for data exploration')">Jupyter</button>
    </div>
    <div id="chat" class="chat"></div>
    <div class="row">
      <input id="in" placeholder="Type a message and press Enter" onkeypress="if(event.key==='Enter'){sendMessage()}" />
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>
</body>
</html>
            """
        )


def main() -> None:
    init_db()
    port = int(os.getenv("PORT", "8000"))
    with socketserver.TCPServer(("", port), RockyHandler) as httpd:
        print(f"Rocky running on http://localhost:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()


