#!/usr/bin/env python3
"""Cloudflare KV over the REST API, because one wrangler process per key does not scale.

`npx wrangler kv:key get` costs about a second of process startup before it does any work, so
reading a namespace one key at a time took over forty minutes at 876 keys -- long enough that the
export was abandoned halfway during a live study, which is exactly when the backup matters. The
REST API reads 100 keys per request, so the same namespace comes back in about ten.

Authentication reuses wrangler's own OAuth token (it is already on disk and already scoped for
workers_kv). Those tokens expire after an hour; on a 401 we run one wrangler command, which
refreshes the file, and retry. CLOUDFLARE_API_TOKEN overrides all of this if set.

The namespace is the one named by CF_KV_NAMESPACE unless a caller passes another.
"""
import json, os, re, subprocess, urllib.error, urllib.parse, urllib.request

API = "https://api.cloudflare.com/client/v4"
CONF = os.path.expanduser("~/.config/.wrangler/config/default.toml")


def _token_from_disk():
    m = re.search(r'oauth_token\s*=\s*"([^"]+)"', open(CONF, encoding="utf-8").read())
    if not m:
        raise SystemExit(f"no oauth_token in {CONF}; run `npx wrangler whoami` to log in")
    return m.group(1)


class KV:
    def __init__(self, ns=None, account=None):
        self.ns = ns or os.environ["CF_KV_NAMESPACE"]
        self.token = os.environ.get("CLOUDFLARE_API_TOKEN") or _token_from_disk()
        self._refreshable = "CLOUDFLARE_API_TOKEN" not in os.environ
        self.account = account or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or self._first_account()

    # ---- plumbing -------------------------------------------------------
    def _raw(self, method, path, body=None):
        req = urllib.request.Request(
            API + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}",
                     **({"content-type": "application/json"} if body is not None else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    def _call(self, method, path, body=None):
        try:
            d = self._raw(method, path, body)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and self._refreshable:
                # wrangler rewrites the token file as a side effect of any authenticated command.
                subprocess.run(["npx", "--yes", "wrangler@3.114.1", "whoami"],
                               capture_output=True, text=True)
                self.token = _token_from_disk()
                self._refreshable = False          # one retry, then let it fail loudly
                d = self._raw(method, path, body)
            else:
                raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read()[:400].decode()}")
        if not d.get("success"):
            raise SystemExit(f"{method} {path} -> {d.get('errors')}")
        return d

    def _first_account(self):
        r = self._call("GET", "/accounts")["result"]
        if not r:
            raise SystemExit("this token can see no accounts")
        return r[0]["id"]

    def _ns(self, tail):
        return f"/accounts/{self.account}/storage/kv/namespaces/{self.ns}{tail}"

    # ---- operations -----------------------------------------------------
    def keys(self, prefix=None):
        """Every key name in the namespace, following the cursor."""
        out, cursor = [], None
        while True:
            q = {"limit": "1000"}
            if prefix:
                q["prefix"] = prefix
            if cursor:
                q["cursor"] = cursor
            d = self._call("GET", self._ns("/keys?" + urllib.parse.urlencode(q)))
            out += [k["name"] for k in d["result"]]
            cursor = (d.get("result_info") or {}).get("cursor")
            if not cursor:
                return out

    def get_many(self, names, chunk=100, on_chunk=None):
        """{name: parsed value} for the names given, 100 per request."""
        out = {}
        for i in range(0, len(names), chunk):
            part = names[i:i + chunk]
            d = self._call("POST", self._ns("/bulk/get"), {"keys": part, "type": "json"})
            out.update(d["result"].get("values") or {})
            if on_chunk:
                on_chunk(min(i + chunk, len(names)), len(names))
        return out

    def delete_many(self, names, chunk=10000):
        for i in range(0, len(names), chunk):
            self._call("POST", self._ns("/bulk/delete"), names[i:i + chunk])
        return len(names)

    def put(self, name, value):
        self._call("PUT", self._ns("/bulk"),
                   [{"key": name, "value": json.dumps(value, ensure_ascii=False)}])
