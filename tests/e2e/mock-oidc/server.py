"""Mock OIDC provider for E2E tests.

Generates RSA key pair on startup, serves OIDC discovery + JWKS endpoints,
and issues signed JWTs via POST /token.
"""

import base64
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

ISSUER = "http://mock-oidc:9090"
KID = "test-key-1"
PORT = 9090

# Generate RSA key pair on startup
_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_key = _private_key.public_key()
_private_pem = _private_key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)

# Derive JWKS components
_public_numbers = _public_key.public_numbers()


def _b64url(value: int, length: int) -> str:
    """Encode an integer as base64url without padding."""
    data = value.to_bytes(length, byteorder="big")
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


_n = _b64url(_public_numbers.n, 256)  # 2048 bits = 256 bytes
_e = _b64url(_public_numbers.e, 3)

DISCOVERY = {
    "issuer": ISSUER,
    "jwks_uri": f"{ISSUER}/.well-known/jwks.json",
}

JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "kid": KID,
            "n": _n,
            "e": _e,
        }
    ]
}


class OIDCHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/.well-known/openid-configuration":
            self._json_response(DISCOVERY)
        elif self.path == "/.well-known/jwks.json":
            self._json_response(JWKS)
        elif self.path == "/health":
            self._json_response({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/token":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            sub = body.get("sub", "test-user")
            roles = body.get("roles", [])

            now = int(time.time())
            claims = {
                "sub": sub,
                "aud": "test-audience",
                "iss": ISSUER,
                "roles": roles,
                "iat": now,
                "exp": now + 3600,
            }

            token = jwt.encode(
                claims,
                _private_pem,
                algorithm="RS256",
                headers={"kid": KID},
            )

            self._json_response({"token": token})
        else:
            self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[mock-oidc] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), OIDCHandler)
    print(f"[mock-oidc] Listening on port {PORT}")
    server.serve_forever()
