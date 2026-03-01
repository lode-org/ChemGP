# -- Project information -----------------------------------------------------
project = "ChemGP"
copyright = "2025, Rohit Goswami"
author = "Rohit Goswami"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

html_context = {
    "source_type": "github",
    "source_user": "HaoZeke",
    "source_repo": "ChemGP",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_theme_options = {
    "github_url": "https://github.com/HaoZeke/ChemGP",
    "accent_color": "teal",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "rgpycrumbs",
                    "url": "https://rgpycrumbs.rgoswami.me",
                    "summary": "Analytical and visualization suite for chemical physics",
                    "external": True,
                },
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Long-timescale molecular dynamics engine",
                    "external": True,
                },
                {
                    "title": "rgpot",
                    "url": "https://github.com/HaoZeke/rgpot",
                    "summary": "RPC potential evaluation for GP-accelerated optimization",
                    "external": True,
                },
            ],
        },
        {
            "title": "crates.io",
            "url": "https://crates.io/crates/chemgp-core",
            "external": True,
        },
    ],
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_baseurl = "chemgp.rgoswami.me"
