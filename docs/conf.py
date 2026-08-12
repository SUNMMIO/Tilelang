# General information about the project.
project = "TileLang-Mesh"
author = "SUNMMIO and TileLang-Mesh Contributors"
copyright = f"2025-2026, {author}"

# Version information.
with open("../VERSION") as f:
    version = f.read().strip()
release = version

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_reredirects",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autodoc_typehints = "description"

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["colon_fence", "deflist"]

redirects = {
    "get_started/try_out": "../index.html#getting-started",
    "sunmmio/sunmmio_tilelang_getting_started_en": "sunmmio_tilelang_getting_started.html",
    "sunmmio/sunmmio_tilelang_user_guide_en": "sunmmio_tilelang_user_guide.html",
}

language = "en"

exclude_patterns = ["_build", "autoapi", "Thumbs.db", ".DS_Store", "README.md", "**/*libinfo*", "**/*version*"]

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
templates_path = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]
footer_copyright = "Copyright 2025-2026 SUNMMIO and TileLang-Mesh Contributors"
footer_note = " "

html_theme_options = {"light_logo": "img/logo-v2.png", "dark_logo": "img/logo-v2.png"}

header_links = [
    ("Home", "https://github.com/SUNMMIO/Tilelang"),
    ("GitHub", "https://github.com/SUNMMIO/Tilelang"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "SUNMMIO",
    "github_repo": "Tilelang",
    "github_version": "tilelang_mesh_main/docs/",
    "theme_vcs_pageview_mode": "edit",
}
