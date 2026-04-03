from __future__ import annotations

import threading
import time


class RelayController:
    """
    Controls an access relay / door-lock solenoid.

    On Raspberry Pi: drives a real BCM GPIO pin via RPi.GPIO.
    On any other platform (Windows, Linux dev machine, etc.):
    prints simulated state changes to stdout so the rest of the
    code path can be exercised without hardware.

    Usage:
        relay = RelayController(pin=18, active_high=True)
        relay.open_door(duration_sec=2.0)   # non-blocking pulse
        relay.cleanup()                      # call on shutdown
    """

    def __init__(
        self,
        pin: int = 18,
        active_high: bool = True,
        door_open_sec: float = 2.0,
    ) -> None:
        self.pin = pin
        self.active_high = active_high
        self.door_open_sec = door_open_sec
        self._lock = threading.Lock()
        self._gpio: object = None
        self._simulation = False

        try:
            import RPi.GPIO as GPIO  # type: ignore[import-not-found]
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            # Safe initial state: relay closed
            GPIO.output(self.pin, GPIO.LOW if self.active_high else GPIO.HIGH)
            self._gpio = GPIO
            print(f"[relay] GPIO ready, pin={self.pin}")
        except Exception as exc:
            self._simulation = True
            print(f"[relay] GPIO unavailable ({exc.__class__.__name__}: {exc}), simulation mode")

    @property
    def is_simulated(self) -> bool:
        return self._simulation

    def open_door(self, duration_sec: float | None = None) -> None:
        """Activate relay for *duration_sec* seconds (non-blocking)."""
        secs = duration_sec if duration_sec is not None else self.door_open_sec
        threading.Thread(
            target=self._pulse, args=(max(0.1, secs),), daemon=True
        ).start()

    def _pulse(self, duration_sec: float) -> None:
        with self._lock:
            self._set_active(True)
            time.sleep(duration_sec)
            self._set_active(False)

    def _set_active(self, active: bool) -> None:
        if self._gpio is not None:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import-not-found]
                if self.active_high:
                    level = GPIO.HIGH if active else GPIO.LOW
                else:
                    level = GPIO.LOW if active else GPIO.HIGH
                GPIO.output(self.pin, level)
            except Exception:
                pass
        state = "OPEN" if active else "CLOSED"
        if self._simulation:
            print(f"[relay] pin={self.pin} -> {state}")

    def cleanup(self) -> None:
        if self._gpio is not None:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import-not-found]
                self._set_active(False)
                GPIO.cleanup()
            except Exception:
                pass
