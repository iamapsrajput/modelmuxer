# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""API route modules.

Route handlers resolve shared singletons (router, db, model_muxer, settings)
through the ``app.main`` module at call time so tests can keep patching
``app.main.<name>``.
"""
