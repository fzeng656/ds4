#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


MAX_BODY = 4 * 1024 * 1024
MAX_PROMPT_TOKENS = 65536
MAX_OUTPUT_TOKENS = 2048


class WorkerError(RuntimeError):
    pass


class NativeWorker:
    def __init__(self, worker, model, manifest, ngram):
        env = os.environ.copy()
        env["QN_RUNTIME_MODE"] = "stable"
        self.proc = subprocess.Popen(
            [str(worker), str(model), str(manifest), str(ngram)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.busy = False
        self.fatal = None
        self.cached_tokens = 0
        self.last_reused_tokens = 0
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY stable "):
            err = self.proc.stderr.read().strip() if self.proc.poll() is not None else ""
            self.proc.kill()
            raise WorkerError(f"worker did not become ready: {ready!r} {err}")
        self.ready_line = ready
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()

    def _drain_stderr(self):
        for line in self.proc.stderr:
            sys.stderr.write(f"[qn-worker] {line}")
            sys.stderr.flush()

    def healthy(self):
        return self.proc.poll() is None and self.fatal is None

    def status(self):
        with self.state_lock:
            return {"healthy": self.healthy(), "busy": self.busy, "fatal": self.fatal, "ready": self.ready_line,
                    "cached_tokens": self.cached_tokens, "last_reused_tokens": self.last_reused_tokens}

    def iter_generate(self, prompt_ids, max_tokens, stop_ids):
        if not self.healthy():
            raise WorkerError(self.fatal or f"worker exited rc={self.proc.poll()}")
        if not 1 <= len(prompt_ids) <= MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt token count must be 1..{MAX_PROMPT_TOKENS}")
        if not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_tokens must be 1..{MAX_OUTPUT_TOKENS}")
        if len(prompt_ids) + max_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt + generation exceeds {MAX_PROMPT_TOKENS}-token production context")
        stop_ids = list(dict.fromkeys(int(x) for x in stop_ids))[:32]
        fields = ["GEN", str(max_tokens), str(len(stop_ids))]
        fields += [str(x) for x in stop_ids]
        fields += [str(len(prompt_ids))]
        fields += [str(int(x)) for x in prompt_ids]
        command = " ".join(fields) + "\n"
        with self.lock:
            with self.state_lock:
                self.busy = True
            try:
                self.proc.stdin.write(command)
                self.proc.stdin.flush()
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise WorkerError(f"worker EOF rc={self.proc.poll()}")
                    line = line.rstrip("\r\n")
                    if line.startswith("ERR "):
                        raise ValueError(line)
                    if line.startswith("FATAL "):
                        with self.state_lock:
                            self.fatal = line
                        raise WorkerError(line)
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "BEGIN" and len(parts) in (3, 4):
                        reused = int(parts[3]) if len(parts) == 4 else 0
                        with self.state_lock:
                            self.last_reused_tokens = reused
                        yield {"type": "begin", "prompt_tokens": int(parts[1]), "prefill_ms": float(parts[2]), "reused_tokens": reused}
                    elif parts[0] == "TOK" and len(parts) == 3:
                        yield {"type": "token", "id": int(parts[1]), "decode_ms": float(parts[2])}
                    elif parts[0] == "END" and len(parts) == 4:
                        completion = int(parts[2])
                        with self.state_lock:
                            self.cached_tokens = min(MAX_PROMPT_TOKENS, len(prompt_ids) + completion)
                        yield {"type": "end", "finish_reason": parts[1], "completion_tokens": completion, "decode_ms": float(parts[3])}
                        break
                    else:
                        raise WorkerError(f"invalid worker response: {line}")
            finally:
                with self.state_lock:
                    self.busy = False

    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            with self.lock:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.stdout.readline()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


class App:
    def __init__(self, args):
        self.args = args
        self.model = pathlib.Path(args.model).expanduser().resolve()
        self.manifest = pathlib.Path(args.manifest).expanduser().resolve()
        self.ngram = pathlib.Path(args.ngram or self.model / "ngram_table.bin").expanduser().resolve()
        self.worker_path = pathlib.Path(args.worker).expanduser().resolve()
        self.model_id = args.model_id or self.model.name
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model), trust_remote_code=True, local_files_only=True)
        self.stop_ids = self._default_stop_ids()
        self.worker = NativeWorker(self.worker_path, self.model, self.manifest, self.ngram)
        self.slots = threading.BoundedSemaphore(args.max_queue + 1)

    def _default_stop_ids(self):
        ids = set()
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int) and eos >= 0:
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(int(x) for x in eos if isinstance(x, int) and x >= 0)
        unk = self.tokenizer.unk_token_id
        for tok in ("<|im_end|>", "<|endoftext|>"):
            try:
                i = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(i, int) and i >= 0 and i != unk:
                    ids.add(i)
            except Exception:
                pass
        return sorted(ids)

    def encode_completion(self, prompt):
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        if isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
            return prompt
        raise ValueError("prompt must be a string or token-id array")

    def encode_chat(self, messages, tools=None):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if tools is not None:
            kwargs["tools"] = tools
        ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(ids, dict):
            ids = ids.get("input_ids")
        elif hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if ids is None:
            raise ValueError("chat template did not return input_ids")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("batched chat templates are not supported")
            ids = ids[0]
        return [int(x) for x in ids]

    def validate_generation(self, body):
        n = body.get("n", 1)
        if n != 1:
            raise ValueError("only n=1 is supported")
        temp = body.get("temperature", 0)
        if temp is None:
            temp = 0
        if float(temp) != 0.0:
            raise ValueError("stable server currently supports greedy generation only (temperature=0)")
        max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 16))
        max_tokens = int(max_tokens)
        if not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_tokens must be 1..{MAX_OUTPUT_TOKENS}")
        return max_tokens


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen-native/phase4i"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    def _auth_ok(self):
        key = self.app.args.api_key
        if not key:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {key}"

    def _send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, message, typ="invalid_request_error"):
        self._send_json(code, {"error": {"message": str(message), "type": typ, "param": None, "code": None}})

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid Content-Length")
        if n <= 0 or n > MAX_BODY:
            raise ValueError(f"request body must be 1..{MAX_BODY} bytes")
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"invalid JSON: {exc}")
        if not isinstance(obj, dict):
            raise ValueError("JSON body must be an object")
        return obj

    def do_GET(self):
        if not self._auth_ok():
            return self._error(HTTPStatus.UNAUTHORIZED, "invalid API key", "authentication_error")
        path = urlparse(self.path).path
        if path in ("/health", "/healthz"):
            st = self.app.worker.status()
            return self._send_json(HTTPStatus.OK if st["healthy"] else HTTPStatus.SERVICE_UNAVAILABLE,
                                   {"status": "ok" if st["healthy"] else "error", "worker": st,
                                    "model": self.app.model_id, "stop_ids": self.app.stop_ids, "context_limit": MAX_PROMPT_TOKENS, "prefill_chunk": 2048})
        if path == "/v1/models":
            return self._send_json(HTTPStatus.OK, {"object": "list", "data": [{"id": self.app.model_id, "object": "model", "created": 0, "owned_by": "local"}]})
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self):
        if not self._auth_ok():
            return self._error(HTTPStatus.UNAUTHORIZED, "invalid API key", "authentication_error")
        path = urlparse(self.path).path
        if path not in ("/v1/completions", "/v1/chat/completions"):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, exc)
        if not self.app.slots.acquire(blocking=False):
            return self._error(HTTPStatus.TOO_MANY_REQUESTS, "native worker queue is full", "rate_limit_error")
        try:
            try:
                max_tokens = self.app.validate_generation(body)
                if path == "/v1/chat/completions":
                    prompt_ids = self.app.encode_chat(body.get("messages"), body.get("tools"))
                    kind = "chat"
                else:
                    prompt_ids = self.app.encode_completion(body.get("prompt"))
                    kind = "completion"
                if not 1 <= len(prompt_ids) <= MAX_PROMPT_TOKENS:
                    raise ValueError(f"prompt token count {len(prompt_ids)} exceeds stable limit {MAX_PROMPT_TOKENS}")
                stream = bool(body.get("stream", False))
                if stream:
                    return self._stream(kind, body, prompt_ids, max_tokens)
                return self._complete(kind, body, prompt_ids, max_tokens)
            except ValueError as exc:
                return self._error(HTTPStatus.BAD_REQUEST, exc)
            except WorkerError as exc:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, exc, "server_error")
            except Exception as exc:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc, "server_error")
        finally:
            self.app.slots.release()

    def _run_collect(self, prompt_ids, max_tokens):
        visible = []
        begin = None
        end = None
        stopset = set(self.app.stop_ids)
        for ev in self.app.worker.iter_generate(prompt_ids, max_tokens, self.app.stop_ids):
            if ev["type"] == "begin": begin = ev
            elif ev["type"] == "token":
                if ev["id"] not in stopset: visible.append(ev["id"])
            elif ev["type"] == "end": end = ev
        if begin is None or end is None:
            raise WorkerError("worker response missing BEGIN/END")
        return visible, begin, end

    def _complete(self, kind, body, prompt_ids, max_tokens):
        out_ids, begin, end = self._run_collect(prompt_ids, max_tokens)
        text = self.app.tokenizer.decode(out_ids, skip_special_tokens=True)
        cid = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": end["completion_tokens"], "total_tokens": len(prompt_ids) + end["completion_tokens"]}
        if kind == "chat":
            obj = {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": self.app.model_id,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": end["finish_reason"]}], "usage": usage,
                   "native_timing": {"prefill_ms": begin["prefill_ms"], "decode_ms": end["decode_ms"], "reused_tokens": begin.get("reused_tokens", 0)}}
        else:
            obj = {"id": cid, "object": "text_completion", "created": int(time.time()), "model": self.app.model_id,
                   "choices": [{"index": 0, "text": text, "finish_reason": end["finish_reason"], "logprobs": None}], "usage": usage,
                   "native_timing": {"prefill_ms": begin["prefill_ms"], "decode_ms": end["decode_ms"], "reused_tokens": begin.get("reused_tokens", 0)}}
        return self._send_json(HTTPStatus.OK, obj)

    def _sse(self, obj):
        data = "data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, separators=(",", ":"))) + "\n\n"
        self.wfile.write(data.encode("utf-8")); self.wfile.flush()

    def _safe_sse(self, obj):
        try:
            self._sse(obj)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return False

    def _stream(self, kind, body, prompt_ids, max_tokens):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        cid = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex
        created = int(time.time()); visible = []; sent = ""; stopset = set(self.app.stop_ids); end = None
        client_ok = True
        if kind == "chat":
            client_ok = self._safe_sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.app.model_id,
                       "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        try:
            # Even after a client disconnect, keep draining the worker through END.
            # Otherwise the next request would consume stale token lines.
            for ev in self.app.worker.iter_generate(prompt_ids, max_tokens, self.app.stop_ids):
                if ev["type"] == "token" and ev["id"] not in stopset:
                    visible.append(ev["id"])
                    if client_ok:
                        now = self.app.tokenizer.decode(visible, skip_special_tokens=True)
                        delta = now[len(sent):] if now.startswith(sent) else self.app.tokenizer.decode([ev["id"]], skip_special_tokens=True)
                        sent = now
                        if delta:
                            if kind == "chat":
                                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.app.model_id,
                                       "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                            else:
                                obj = {"id": cid, "object": "text_completion", "created": created, "model": self.app.model_id,
                                       "choices": [{"index": 0, "text": delta, "finish_reason": None, "logprobs": None}]}
                            client_ok = self._safe_sse(obj)
                elif ev["type"] == "end":
                    end = ev
        except WorkerError as exc:
            if client_ok:
                self._safe_sse({"error": {"message": str(exc), "type": "server_error"}})
                self._safe_sse("[DONE]")
            self.close_connection = True
            return
        reason = end["finish_reason"] if end else "error"
        if client_ok:
            if kind == "chat":
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.app.model_id,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
            else:
                obj = {"id": cid, "object": "text_completion", "created": created, "model": self.app.model_id,
                       "choices": [{"index": 0, "text": "", "finish_reason": reason, "logprobs": None}]}
            client_ok = self._safe_sse(obj)
            if client_ok:
                self._safe_sse("[DONE]")
        self.close_connection = True


class NativeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32


def is_loopback(host):
    return host in ("127.0.0.1", "::1", "localhost")


def main():
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Stable OpenAI-compatible HTTP front-end for qwen-native")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ngram")
    ap.add_argument("--worker", default=str(here / "qn_worker"))
    ap.add_argument("--model-id")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--api-key", default=os.getenv("QN_API_KEY"))
    ap.add_argument("--max-queue", type=int, default=4, help="waiting requests allowed in addition to the active request")
    args = ap.parse_args()
    if not is_loopback(args.host) and not args.api_key:
        raise SystemExit("refusing non-loopback bind without --api-key or QN_API_KEY")
    if args.max_queue < 0 or args.max_queue > 64:
        raise SystemExit("--max-queue must be 0..64")
    app = App(args)
    print(f"qwen-native server ready http://{args.host}:{args.port} model={app.model_id} stop_ids={app.stop_ids}", file=sys.stderr)
    httpd = NativeHTTPServer((args.host, args.port), Handler)
    httpd.app = app
    stopping = threading.Event()
    def request_shutdown(signum=None, frame=None):
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        httpd.serve_forever(poll_interval=0.25)
    finally:
        httpd.server_close(); app.worker.close()


if __name__ == "__main__":
    main()
