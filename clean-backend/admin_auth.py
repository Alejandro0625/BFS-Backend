# -*- coding: utf-8 -*-
"""FAIL-CLOSED authentication for the /admin/* surface.

WHY THIS MODULE EXISTS
----------------------
`app.py` used to guard both admin routes with

    if key != os.environ.get("ADMIN_KEY", <a short literal>):

The second argument of `os.environ.get` is not a placeholder -- it is the value the
comparison actually uses whenever the environment variable is unset.  So with no
`ADMIN_KEY` configured, that literal WAS the live credential, and it was published:
the repository is public, the literal sat in `clean-backend/app.py`, and `PLAYBOOK.md`
printed the live Railway URL next to the exact `POST /admin/upload-model?key=...` call
spelling it out.  Measured against production on 2026-08-05, before this module landed,
that key returned **HTTP 200** from `/admin/export-corrections`.  It authenticated:

  * `POST /admin/upload-model`      -- writes the ONNX file every later inference
                                       loads, i.e. silent poisoning of every SF
                                       number the system produces;
  * `GET  /admin/export-corrections` -- dumps every human-captured label, the
                                       training moat.

THE RULE
--------
An unset variable is not a permission to fall back to a public string.  It is the
absence of a decision, and the safe reading of an absent decision is NO.  This module
therefore treats three states as one:

    unset  ==  blank  ==  still the published default   ->  the endpoint is DISABLED

Disabled means every request is refused, including one carrying the published default.
The endpoint becomes reachable only after an operator sets a real `ADMIN_KEY`.  That
cannot break correct usage, because "correct usage" of a world-readable credential is
not a thing that exists.

NO CREDENTIAL LITERAL LIVES HERE
--------------------------------
The burned default is recorded as a SHA-256 digest on a denylist.  A digest cannot be
sent to the server as a key, and the string it fingerprints is already world-readable
in the repository's own history, so recording it costs nothing and buys the one thing
that matters: the ability to keep refusing that exact value forever, even if someone
later pastes it into the Railway dashboard because an old playbook told them to.
"""
import hashlib
import hmac
import logging
import os

log = logging.getLogger("bfs.admin_auth")

ENV_VAR = "ADMIN_KEY"

# SHA-256 of credentials that are BURNED: values that were published in the public
# repository and can therefore never authenticate anything again.  Denylist only --
# nothing here is ever used to grant access.
#   a5e02022... = the former in-code default of ADMIN_KEY, printed verbatim in
#                 PLAYBOOK.md next to the live service URL.
_BURNED_DIGESTS = frozenset({
    "a5e02022e493ebc2bf6d2d31247886f9d0465b4656485839ace9d13e8f474a3f",
})

# Client-facing text.  Deliberately explicit: the operator is the person most likely
# to hit this, and a vague 403 would send them hunting for a key that does not exist.
_DISABLED_MSG = (
    "admin endpoints are DISABLED: no ADMIN_KEY is configured on this service, or the "
    "configured value is the credential that was published in the public repository. "
    "Set ADMIN_KEY to a fresh secret in the service environment and redeploy; until "
    "then this endpoint refuses every request, including one presenting the published "
    "default."
)

_MIN_RECOMMENDED_LEN = 16

# One WARNING per process per reason -- a disabled endpoint under a scan must not be
# able to flood the log, but the operator must still see the reason at least once.
_warned = set()


def _warn_once(tag, msg):
    if tag not in _warned:
        _warned.add(tag)
        log.warning(msg)


def _configured():
    return (os.environ.get(ENV_VAR) or "").strip()


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def status():
    """(enabled, reason) -- is the admin surface usable at all, and why not.

    Pure and side-effect free apart from the one-shot operator warning, so tests and
    /health can both call it without arming anything.
    """
    key = _configured()
    if not key:
        _warn_once("unset",
                   "ADMIN_KEY is not set -- /admin/* is DISABLED and refuses every "
                   "request. This is the fail-closed default; set ADMIN_KEY to a fresh "
                   "secret to enable the admin surface.")
        return False, "ADMIN_KEY is not set"
    if _digest(key) in _BURNED_DIGESTS:
        _warn_once("burned",
                   "ADMIN_KEY is set to a credential that was PUBLISHED in the public "
                   "repository -- /admin/* stays DISABLED. Rotate it to a value that "
                   "has never appeared in source or documentation.")
        return False, "ADMIN_KEY is a published, burned credential"
    if len(key) < _MIN_RECOMMENDED_LEN:
        # A short key is weak, not burned. Warn; do NOT refuse -- inventing a new
        # lockout the operator never agreed to is its own outage.
        _warn_once("short",
                   "ADMIN_KEY is shorter than %d characters. It is accepted, but a "
                   "guessable admin key on a public URL is worth very little."
                   % _MIN_RECOMMENDED_LEN)
    return True, "ok"


def authorize(supplied, endpoint="/admin/*"):
    """(ok, http_status, message) for one admin request.

    503 -- the endpoint is administratively disabled (fail-closed). It is not that the
           caller got the key wrong; there is no key that opens this door right now.
    403 -- a real key is configured and the caller did not present it.
    """
    enabled, reason = status()
    if not enabled:
        log.error("REFUSED %s: %s. No request can authenticate while this holds.",
                  endpoint, reason)
        return False, 503, _DISABLED_MSG

    # Constant-time: the old `key != os.environ.get(...)` compared with `!=`, which
    # returns as soon as two bytes differ and leaks the shared prefix length by timing.
    if not supplied or not hmac.compare_digest(str(supplied), _configured()):
        log.warning("REFUSED %s: bad admin key presented (%d chars).",
                    endpoint, len(str(supplied or "")))
        return False, 403, "bad key"
    return True, 200, "ok"
