"""Documentation-only build; no scientific-runtime imports."""
import os

project = 'Calcium video simulator'
author = "Neurallabware"
extensions = ["myst_parser", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = project
myst_enable_extensions = ["colon_fence", "deflist"]
html_theme_options = {
    "source_repository": "https://github.com/Neurallabware/Simulate_Ca_Imaging_video/",
    "source_branch": (os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main") if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external" else os.environ.get("READTHEDOCS_GIT_IDENTIFIER") or os.environ.get("GITHUB_HEAD_REF") or "main"),
    "source_directory": "docs/",
}
