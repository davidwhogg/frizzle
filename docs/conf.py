import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))

project = "frizzle"
author = "David W. Hogg & Andy Casey"
copyright = "2024-2026, David W. Hogg & Andy Casey"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "dark_plot",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "frizzle"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_sidebars = {"**": []}
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#b03030",
        "color-brand-content": "#b03030",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff7a7a",
        "color-brand-content": "#ff7a7a",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/andycasey/frizzle",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}
pygments_style = "tango"
pygments_dark_style = "monokai"

# matplotlib plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 144)]
plot_rcparams = {
    "savefig.bbox": "tight",
    "figure.autolayout": True,
    "font.size": 14,
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}
plot_apply_rcparams = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
}
autodoc_typehints = "description"
napoleon_use_rtype = False
