"""Knowing when JARVIS is done, without asking the console.

V1 is a desktop application that the user starts by hand and stops when they are
finished. Until this module the only way to stop it was Ctrl+C in the launcher's
console window, which quietly made the terminal part of normal use -- the exact
thing the desktop shell exists to avoid.

There are two ways a person closes a window, and this handles the second one.

The first is deliberate: the **Kapat** button on the page, which asks the service
to stop and gets an answer before the socket goes away.

The second is what people actually do: close the browser tab. Nothing is sent
then, and a process that only listened for the button would sit there afterwards
as a daemon nobody asked for. So the server's client count is watched instead:
when the last client leaves and none comes back, JARVIS stops on its own.

## Why a grace period rather than an instant stop

Reloading the page disconnects too. So does a laptop lid, a sleeping wifi
adapter, a browser tab discarded to save memory. Treating any of those as "the
user is finished" would make JARVIS die under the user's hands and look like a
crash. The grace period is the whole safety margin: long enough that a reload
never reaches it, short enough that a closed tab does not leave a process
running for the rest of the day.

`OrphanWatch` is pure and takes its clock as an argument, so all of that is
testable without waiting two minutes for anything.
"""

from __future__ import annotations

import logging

log = logging.getLogger("jarvis.cli.lifecycle")

#: Long enough to outlast a page reload on a slow machine, short enough that a
#: closed tab does not leave a process running all afternoon.
DEFAULT_GRACE_S = 120.0


class OrphanWatch:
    """Decides whether JARVIS has been left alone long enough to stop.

    Feed it the current client count and the current time; it answers whether
    the process should now shut down. It deliberately does nothing until the
    first client has connected: at startup there is no browser attached yet, and
    a watch that started counting immediately would close JARVIS before the page
    it just opened had finished loading.
    """

    __slots__ = ("grace_s", "_seen_client", "deadline")

    def __init__(self, *, grace_s: float = DEFAULT_GRACE_S) -> None:
        #: Zero (or less) switches the watch off entirely, for someone who wants
        #: JARVIS to outlive the page on purpose.
        self.grace_s = float(grace_s)
        self._seen_client = False
        #: When the countdown expires, or None when nothing is counting.
        self.deadline: float | None = None

    def observe(self, clients: int, *, now: float) -> bool:
        """Returns True when the process should stop."""
        if self.grace_s <= 0:
            return False
        if clients > 0:
            self._seen_client = True
            self.deadline = None
            return False
        if not self._seen_client:
            # Nobody has ever connected. Still starting up, not abandoned.
            return False
        if self.deadline is None:
            self.deadline = now + self.grace_s
            log.debug("istemci kalmadı, %.0f sn sonra kapanılacak", self.grace_s)
            return False
        return now > self.deadline


__all__ = ["OrphanWatch", "DEFAULT_GRACE_S"]
