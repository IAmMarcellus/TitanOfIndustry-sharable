# Marks `proxy/` as a package so the LiteLLM proxy can import `proxy.hooks.proxy_handler_instance`
# (the vision-wake pre-call hook). `make proxy` runs litellm with PYTHONPATH set to the repo root.
