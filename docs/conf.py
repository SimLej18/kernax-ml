from __future__ import annotations

project = "kernax"
author = "S. Lejoly"
release = "0.5.5-alpha"

extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.napoleon",
	"sphinx.ext.viewcode",
	"sphinx_autodoc_typehints",
	"myst_nb",
]

html_theme = "shibuya"
html_theme_options = {
	"github_url": "https://github.com/SimLej18/kernax-ml",
}

# Jupytext: use .py files as notebook sources
nb_custom_formats = {
	".py": ["jupytext.reads", {"fmt": "py:percent"}]
}

# Don't execute notebooks during build for now
nb_execution_mode = "off"

# NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_member_order = "bysource"
exclude_patterns = ["_build", ".jupyter_cache"]
