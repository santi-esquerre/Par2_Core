# -- Par2_Core Sphinx Configuration ----------------------------------------
# Full docs: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

project = "Par2_Core"
copyright = "2026, Par2_Core contributors"
author = "Par2_Core contributors"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.mathjax",  # LaTeX math rendering
    "sphinx.ext.todo",  # TODO directives
    "sphinx.ext.githubpages",  # .nojekyll for GitHub Pages
    "breathe",  # Doxygen → Sphinx bridge
    "myst_parser",  # Markdown support (.md files)
]

# ---------------------------------------------------------------------------
# Breathe (Doxygen ↔ Sphinx)
# ---------------------------------------------------------------------------
breathe_projects = {
    "par2_core": os.path.abspath("../_doxygen/xml"),
}
breathe_default_project = "par2_core"
breathe_default_members = ("members", "undoc-members")

# ---------------------------------------------------------------------------
# MyST (Markdown support)
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",  # $inline$ and $$display$$ math
    "colon_fence",  # ::: directives
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# MathJax
# ---------------------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
}
html_static_path = ["_static"]
html_title = "Par2_Core Documentation"
html_baseurl = "https://santi-esquerre.github.io/Par2_Core/"

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Don't fail on missing Doxygen XML (allows partial builds during dev)
breathe_projects_source = {}

# Suppress duplicate C++ declaration warnings from Breathe
# (all headers share the par2 namespace, causing expected duplicates)
suppress_warnings = ["cpp.duplicate_declaration"]

# Number figures/tables
numfig = True

# TODO extension
todo_include_todos = True
