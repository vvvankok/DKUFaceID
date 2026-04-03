from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from uuid import uuid4

_RETRY_DELAYS = (1.0, 2.0)  # seconds between retries on transient errors


@dataclass
class TelegramAlerter:
    bot_token: str
    chat_id: str
    timeout_sec: float = 5.0

    def is_configured(self) -> bool:
        return bool(self.bot_token.strip() and self.chat_id.strip())

    def send(self, message: str, parse_mode: str = "") -> None:
        if not self.is_configured():
            return
        params: dict[str, str] = {"chat_id": self.chat_id, "text": message}
        if parse_mode:
            params["parse_mode"] = parse_mode
        self._api_get("sendMessage", params)

    def send_html(self, message: str) -> None:
        """Send a message with HTML formatting (bold, italic, code tags)."""
        self.send(message, parse_mode="HTML")

    def send_with_actions(
        self,
        message: str,
        actions: list[tuple[str, str]],
        parse_mode: str = "",
    ) -> None:
        if not self.is_configured():
            return
        keyboard = {
            "inline_keyboard": [[{"text": text, "callback_data": data} for text, data in actions]]
        }
        params: dict[str, str] = {
            "chat_id": self.chat_id,
            "text": message,
            "reply_markup": json.dumps(keyboard, ensure_ascii=False),
        }
        if parse_mode:
            params["parse_mode"] = parse_mode
        self._api_get("sendMessage", params)

    def send_photo(self, image_bytes: bytes, caption: str = "", parse_mode: str = "") -> None:
        if not self.is_configured():
            return
        self._send_photo_core(
            image_bytes=image_bytes, caption=caption, actions=None, parse_mode=parse_mode
        )

    def send_photo_with_actions(
        self,
        image_bytes: bytes,
        caption: str,
        actions: list[tuple[str, str]],
        parse_mode: str = "",
    ) -> None:
        if not self.is_configured():
            return
        self._send_photo_core(
            image_bytes=image_bytes, caption=caption, actions=actions, parse_mode=parse_mode
        )

    def get_updates(
        self,
        offset: int | None = None,
        timeout: int = 10,
        allowed_updates: list[str] | None = None,
    ) -> dict:
        params: dict[str, str] = {"timeout": str(timeout)}
        if offset is not None:
            params["offset"] = str(offset)
        if allowed_updates is not None:
            params["allowed_updates"] = json.dumps(allowed_updates, ensure_ascii=False)
        return self._api_get("getUpdates", params)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        params = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text
        self._api_get("answerCallbackQuery", params)

    def _send_photo_core(
        self,
        image_bytes: bytes,
        caption: str,
        actions: list[tuple[str, str]] | None,
        parse_mode: str = "",
    ) -> None:
        reply_markup = None
        if actions:
            keyboard = {
                "inline_keyboard": [[{"text": text, "callback_data": data} for text, data in actions]]
            }
            reply_markup = json.dumps(keyboard, ensure_ascii=False)

        boundary = f"faceid-{uuid4().hex}"
        body = self._multipart_body(
            boundary=boundary,
            chat_id=self.chat_id,
            caption=caption,
            image_bytes=image_bytes,
            reply_markup=reply_markup,
            parse_mode=parse_mode if parse_mode else None,
        )
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.bot_token}/sendPhoto",
            data=body,
            headers={
                "User-Agent": "faceid-alerts/1.0",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        last_exc: Exception | None = None
        for delay in (*_RETRY_DELAYS, None):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec):
                    return
            except Exception as exc:
                last_exc = exc
                if delay is not None:
                    time.sleep(delay)
        if last_exc is not None:
            raise last_exc

    def _api_get(self, method: str, params: dict[str, str]) -> dict:
        url = (
            f"https://api.telegram.org/bot{self.bot_token}/{method}?"
            + urllib.parse.urlencode(params)
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "faceid-alerts/1.0"},
            method="GET",
        )
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(raw)
                except Exception:
                    return {"ok": False, "raw": raw}
            except Exception as exc:
                last_exc = exc
                if delay is not None:
                    time.sleep(delay)
        return {"ok": False, "error": str(last_exc)}

    @staticmethod
    def _multipart_body(
        boundary: str,
        chat_id: str,
        caption: str,
        image_bytes: bytes,
        reply_markup: str | None,
        parse_mode: str | None = None,
    ) -> bytes:
        crlf = b"\r\n"
        chunks: list[bytes] = []

        def add_text(name: str, value: str) -> None:
            chunks.append(f"--{boundary}".encode())
            chunks.append(f'Content-Disposition: form-data; name="{name}"'.encode())
            chunks.append(b"")
            chunks.append(value.encode("utf-8"))

        add_text("chat_id", chat_id)
        if caption:
            add_text("caption", caption)
        if parse_mode:
            add_text("parse_mode", parse_mode)
        if reply_markup:
            add_text("reply_markup", reply_markup)

        chunks.append(f"--{boundary}".encode())
        chunks.append(
            b'Content-Disposition: form-data; name="photo"; filename="event.jpg"'
        )
        chunks.append(b"Content-Type: image/jpeg")
        chunks.append(b"")
        chunks.append(image_bytes)
        chunks.append(f"--{boundary}--".encode())
        chunks.append(b"")
        return crlf.join(chunks)


@dataclass
class MultiTelegramAlerter:
    """
    Broadcasts all alerts to multiple :class:`TelegramAlerter` instances
    (same bot token, different chat IDs).  Presents the same public interface
    as ``TelegramAlerter`` so existing code needs no changes.

    Bot commands (/help, /stats, …) and callback queries are handled only by
    the *first* configured alerter — a single polling loop is sufficient.
    """

    alerters: list[TelegramAlerter]

    def is_configured(self) -> bool:
        return any(a.is_configured() for a in self.alerters)

    # ── broadcast helpers ────────────────────────────────────────────────────
    def _each(self, method: str, *args, **kwargs) -> None:
        for a in self.alerters:
            if a.is_configured():
                try:
                    getattr(a, method)(*args, **kwargs)
                except Exception:
                    pass

    def _first(self, method: str, *args, **kwargs):
        for a in self.alerters:
            if a.is_configured():
                return getattr(a, method)(*args, **kwargs)
        return {}

    # ── alert methods (broadcast) ────────────────────────────────────────────
    def send(self, message: str, parse_mode: str = "") -> None:
        self._each("send", message, parse_mode)

    def send_html(self, message: str) -> None:
        self._each("send_html", message)

    def send_with_actions(
        self, message: str, actions: list[tuple[str, str]], parse_mode: str = ""
    ) -> None:
        self._each("send_with_actions", message, actions, parse_mode)

    def send_photo(self, image_bytes: bytes, caption: str = "", parse_mode: str = "") -> None:
        self._each("send_photo", image_bytes, caption, parse_mode)

    def send_photo_with_actions(
        self,
        image_bytes: bytes,
        caption: str,
        actions: list[tuple[str, str]],
        parse_mode: str = "",
    ) -> None:
        self._each("send_photo_with_actions", image_bytes, caption, actions, parse_mode)

    # ── command/callback methods (first alerter only) ────────────────────────
    def get_updates(
        self,
        offset: int | None = None,
        timeout: int = 10,
        allowed_updates: list[str] | None = None,
    ) -> dict:
        return self._first("get_updates", offset, timeout, allowed_updates)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        self._first("answer_callback_query", callback_query_id, text)
