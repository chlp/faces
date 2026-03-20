"""Face tracking state machine."""

import time

from config import Config, DEBUG_INTERVAL, UNKNOWN_LABEL


class FaceTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detected: list = []
        self.face_has_stranger = False
        self.face_stranger_conf = False
        self.face_has_known = False
        self.last_greeted: dict = {}
        self._confirm_streak: dict = {}
        self._stranger_streak = 0
        self._last_stranger_ts = 0.0
        self._last_event_time: dict = {}
        self._last_state: tuple = (frozenset(), 0)
        self._last_heartbeat = 0.0
        self._pending_stranger = None
        self._frame_count = 0

    def update(self, face_results, frame) -> list:
        events = []
        self._frame_count += 1

        if face_results is not None:
            self.detected = face_results
            current_names: set = set()
            raw_unk = 0

            for bbox, name, score in face_results:
                if name == UNKNOWN_LABEL:
                    raw_unk += 1
                    continue
                current_names.add(name)
                self._confirm_streak[name] = (
                    self._confirm_streak.get(name, 0) + 1
                )
                if self.cfg.debug:
                    print(
                        f"[D]   streak={self._confirm_streak[name]}"
                        f"/{self.cfg.confirm_frames}"
                    )
                if self._confirm_streak[name] >= self.cfg.confirm_frames:
                    now = time.time()
                    if now - self.last_greeted.get(name, 0) > self.cfg.greet_cooldown:
                        self.last_greeted[name] = now
                        events.append(("greet", name))

            # stranger streak (rate-limited)
            now_ts = time.time()
            if raw_unk > 0:
                if self._stranger_streak == 0 or (
                    now_ts - self._last_stranger_ts
                ) >= self.cfg.streak_min_interval:
                    self._stranger_streak += 1
                    self._last_stranger_ts = now_ts
                if self.cfg.debug:
                    print(
                        f"[D]   stranger_streak="
                        f"{self._stranger_streak}/{self.cfg.confirm_frames}"
                    )
            else:
                self._stranger_streak = 0
            unk_count = (
                raw_unk
                if self._stranger_streak >= self.cfg.confirm_frames
                else 0
            )

            self.face_has_stranger = raw_unk > 0
            self.face_stranger_conf = unk_count > 0
            self.face_has_known = bool(current_names)

            for gone in set(self._confirm_streak) - current_names - {UNKNOWN_LABEL}:
                self._confirm_streak[gone] = 0

            cur_state = (frozenset(current_names), unk_count)
            if cur_state != self._last_state:
                web_names = list(current_names) + [UNKNOWN_LABEL] * unk_count
                if web_names:
                    now = time.time()
                    key = cur_state
                    if current_names:
                        self._pending_stranger = None
                        if now - self._last_event_time.get(key, 0) >= self.cfg.web_event_cooldown:
                            self._last_event_time[key] = now
                            events.append((
                                "web_event", web_names,
                                frame.copy(), list(self.detected),
                            ))
                    else:
                        if self._pending_stranger is None:
                            self._pending_stranger = (
                                frame.copy(), list(self.detected),
                                now, web_names, key,
                            )
                            if self.cfg.debug:
                                print(
                                    f"[D] {time.strftime('%H:%M:%S')} "
                                    f"stranger pending "
                                    f"({self.cfg.stranger_confirm_delay:.0f}s)..."
                                )
                else:
                    self._pending_stranger = None
                    if self.cfg.debug:
                        print(
                            f"[D] {time.strftime('%H:%M:%S')} "
                            f"frame {self._frame_count}: no faces"
                        )
                self._last_state = cur_state
                self._last_heartbeat = time.time()
            elif (
                not current_names
                and raw_unk > 0
                and self._pending_stranger is None
            ):
                now = time.time()
                pn = [UNKNOWN_LABEL] * raw_unk
                pk = (frozenset(), raw_unk)
                self._pending_stranger = (
                    frame.copy(), list(self.detected), now, pn, pk,
                )
                if self.cfg.debug:
                    print(
                        f"[D] {time.strftime('%H:%M:%S')} "
                        f"stranger (pre-streak) pending..."
                    )
            elif (
                self.cfg.debug
                and not current_names
                and not unk_count
                and time.time() - self._last_heartbeat >= DEBUG_INTERVAL
            ):
                print(
                    f"[D] {time.strftime('%H:%M:%S')} "
                    f"frame {self._frame_count}: no faces"
                )
                self._last_heartbeat = time.time()

        # stranger timer (every frame)
        if self._pending_stranger is not None:
            sf, sd, st, sn, sk = self._pending_stranger
            if time.time() - st >= self.cfg.stranger_confirm_delay:
                self._pending_stranger = None
                now = time.time()
                if now - self._last_event_time.get(sk, 0) >= self.cfg.web_event_cooldown:
                    self._last_event_time[sk] = now
                    events.append(("web_event", sn, sf, sd))
                if self.cfg.debug:
                    print(
                        f"[D] {time.strftime('%H:%M:%S')} "
                        f"stranger confirmed -> event"
                    )

        return events
