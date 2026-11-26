"""Persistence layer: where controller state is read from and written to.

Today this is JSON files under data/sessions/ (see session_store.py); the
docstrings note Redis as the eventual backend. Keeping it behind this package
means the rest of the app never touches the filesystem directly.
"""
