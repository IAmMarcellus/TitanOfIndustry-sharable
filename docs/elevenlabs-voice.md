# Mergatroid Voice

This interview snapshot intentionally redacts public-ingress setup, private
hostnames, route mappings, and local port assignments for the cloud voice path.

The retained code shows the shape of the integration:

- Paperclip mints a short-lived voice session for the browser.
- Mergatroid uses a custom LLM path to call the local oversight brain.
- The self-hosted Pipecat path remains available through `make voice`.
- Local endpoint values belong in `.env`, which is gitignored.

The default voice-agent configuration is designed for a local LLM route and will
not work out of the box without one.
