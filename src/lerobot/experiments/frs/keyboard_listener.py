#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Keyboard steering state for FRS (paper: W/A/S/D/Q/E translations).

Discrete key presses accumulate a Cartesian direction that the steering loop
consumes at each planning step. Built on `lerobot.utils.keyboard_input`
(pynput global listener on X11/WSLg, cbreak terminal reader on Wayland/SSH),
so it works in the same environments as the recording hotkeys.

Keys (case-insensitive):
    w / s   +x / -x      a / d   -y / +y      q / e   -z / +z
    g       toggle gripper
    space   defer to the base policy for this planning step
    esc / x quit
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from lerobot.utils.keyboard_input import create_key_listener

_KEY_TO_DIRECTION = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, -1.0, 0.0),
    "d": (0.0, 1.0, 0.0),
    "q": (0.0, 0.0, -1.0),
    "e": (0.0, 0.0, 1.0),
}


class SteeringKeyboardState:
    """Thread-safe accumulation of steering key presses.

    The listener thread only touches plain attributes (GIL-atomic), and
    `consume()` runs on the control thread; a lock guards the reset.
    """

    def __init__(self):
        """Zero-initialized steering state; call `start()` to begin listening."""
        self._lock = threading.Lock()
        self._delta = np.zeros(3, dtype=np.float32)
        self.gripper_open = True
        self.defer_requested = False
        self.quit = False
        self.listener = None

    def start(self):
        """Start the platform-appropriate key listener (see `create_key_listener`)."""
        self.listener = create_key_listener(
            self._on_key,
            controls_help="W/A/S/D/Q/E steer, G gripper, Space defer, Esc quit",
        )
        if self.listener is None:
            logging.warning(
                "No keyboard backend available — steering keys disabled; the policy will run unsteered."
            )
        return self.listener

    def stop(self):
        """Stop the listener and restore the terminal (terminal backend)."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _on_key(self, name: str) -> None:
        """Accumulate a direction / toggle state from one key event."""
        key = name.lower()
        if key in _KEY_TO_DIRECTION:
            with self._lock:
                self._delta += np.array(_KEY_TO_DIRECTION[key], dtype=np.float32)
        elif key == "g":
            self.gripper_open = not self.gripper_open
            logging.info("gripper -> %s", "open" if self.gripper_open else "closed")
        elif key == "space":
            self.defer_requested = True
        elif key in ("esc", "x"):
            self.quit = True

    def consume_delta(self) -> np.ndarray:
        """Return and clear the direction accumulated since the last call."""
        with self._lock:
            delta, self._delta = self._delta, np.zeros(3, dtype=np.float32)
        return delta

    def consume_defer(self) -> bool:
        """Return and clear whether Space (defer) was pressed since the last call."""
        if self.defer_requested:
            self.defer_requested = False
            return True
        return False
