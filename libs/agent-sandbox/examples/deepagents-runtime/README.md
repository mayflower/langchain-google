# DeepAgents Runtime Reference

This runtime is a minimal FastAPI reference for the file and command API used by
`k8s-agent-sandbox`.

Contract:

- `POST /execute`
- `POST /upload`
- `GET /download/{path}`
- `GET /list/{path}`
- `GET /exists/{path}`

The runtime uses `/workspace` by default. Set `SANDBOX_RUNTIME_DIR` to override
it. The image runs as a non-root user and intentionally avoids unnecessary
packages.

The application entry point is `app.py`.
