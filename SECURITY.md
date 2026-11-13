# Security Policy

## Supported Versions

Security updates are applied to the latest commit on the `main` branch.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report security issues privately via [GitHub Security Advisories](https://github.com/qilaidev/cleanplate/security/advisories/new). Include:

- A clear description of the vulnerability and potential impact
- Steps to reproduce or a minimal proof of concept
- Any suggested remediation or affected configuration

We will acknowledge receipt within 72 hours and coordinate disclosure once a fix is available.

## Security Model & Hardening Notes

Cleanplate is designed as a self-hosted internal workflow node:

- **API Key Storage**: API keys (`X-API-Key`) are hashed with SHA-256 on the server and never stored in plaintext. Set `DWM_ENV=production` and replace `DWM_DEFAULT_API_KEY` before exposing the service.
- **Storage Isolation**: `input_path` and artifact downloads are strictly confined to `DWM_STORAGE_ROOT`. Path traversal outside the storage root is rejected.
- **Callback SSRF Guard**: Outbound webhook `callback_url` targets are restricted to public `http`/`https` endpoints by default (`DWM_ALLOW_PRIVATE_CALLBACK_URLS=false` blocks `localhost` and private/loopback IP ranges).
- **Webhook Signatures**: When `callback_secret` is provided, callbacks include `X-Timestamp` and `X-Signature` (`HMAC-SHA256("{timestamp}.{body}", secret)`).
- **Authorized Media Only**: Only process media you own or have explicit rights to modify.
