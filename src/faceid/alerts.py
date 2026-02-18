from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class TelegramAlerter:
    bot_token: str
    chat_id: str
    timeout_sec: float = 5.0

    def is_configured(self) -> bool:
        return bool(self.bot_token.strip() and self.chat_id.strip())

    def send(self, message: str) -> None:
        if not self.is_configured():
            return
        self._api_get("sendMessage", {"chat_id": self.chat_id, "text": message})

    def send_with_actions(self, message: str, actions: list[tuple[str, str]]) -> None:
        if not self.is_configured():
            return
        keyboard = {
            "inline_keyboard": [[{"text": text, "callback_data": data} for text, data in actions]]
        }
        self._api_get(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": message,
                "reply_markup": json.dumps(keyboard, ensure_ascii=False),
            },
        )

    def send_photo(self, image_bytes: bytes, caption: str = "") -> None:
        if not self.is_configured():
            return
        self._send_photo_core(image_bytes=image_bytes, caption=caption, actions=None)

    def send_photo_with_actions(
        self,
        image_bytes: bytes,
        caption: str,
        actions: list[tuple[str, str]],
    ) -> None:
        if not self.is_configured():
            return
        self._send_photo_core(image_bytes=image_bytes, caption=caption, actions=actions)

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
        with urllib.request.urlopen(req, timeout=self.timeout_sec):
            pass

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
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            return {"ok": False, "raw": raw}

    @staticmethod
    def _multipart_body(
        boundary: str,
        chat_id: str,
        caption: str,
        image_bytes: bytes,
        reply_markup: str | None,
    ) -> bytes:
        crlf = b"\r\n"
        chunks: list[bytes] = []

        def add_text(name: str, value: str) -> None:
            chunks.append(f"--{boundary}".encode())
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"'.encode()
            )
            chunks.append(b"")
            chunks.append(value.encode("utf-8"))

        add_text("chat_id", chat_id)
        if caption:
            add_text("caption", caption)
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

