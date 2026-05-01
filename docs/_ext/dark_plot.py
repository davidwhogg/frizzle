"""
Render every ``.. plot::`` directive twice — once with the configured light
rcParams, once with a dark overlay — and emit both images using Furo's
``.only-light`` / ``.only-dark`` CSS classes.

The user code runs twice per directive (light, then dark) so that any
matplotlib drawing primitives whose colors depend on rcParams pick up the
right values for each render.
"""

from matplotlib.sphinxext import plot_directive as mpd


DARK_RCPARAMS_DEFAULT = {
    "savefig.facecolor": "none",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": "#cfcfcf",
    "axes.labelcolor": "#e7e7e7",
    "xtick.color": "#cfcfcf",
    "ytick.color": "#cfcfcf",
    "text.color": "#e7e7e7",
    "grid.color": "#444444",
    "legend.edgecolor": "#cfcfcf",
}


# Custom RST template emitted by plot_directive. Matches the structure of
# matplotlib's TEMPLATE but emits two figure directives — one for light, one
# for dark — each tagged with the appropriate Furo CSS class. The original
# :class: option from matplotlib (which carries 'plot-directive') is filtered
# out so it doesn't conflict with our :class: only-light/only-dark.
DUAL_TEMPLATE = mpd._SOURCECODE + """

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      :class: only-light plot-directive
      {% for option in options -%}
      {%- if not option.startswith(':class:') -%}
      {{ option }}
      {% endif -%}
      {% endfor %}

      {{ caption }}

   .. figure:: {{ build_dir }}/{{ img.basename }}-dark.{{ default_fmt }}
      :class: only-dark plot-directive
      {% for option in options -%}
      {%- if not option.startswith(':class:') -%}
      {{ option }}
      {% endif -%}
      {% endfor %}

      {{ caption }}

   {% endfor %}

.. only:: not html

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor -%}

      {{ caption }}
   {% endfor %}
"""


_original_render_figures = mpd.render_figures


def _render_figures_dual(code, code_path, output_dir, output_base, context,
                         function_name, config, context_reset=False,
                         close_figs=False, code_includes=None):
    # Light pass — runs as configured.
    result = _original_render_figures(
        code, code_path, output_dir, output_base, context,
        function_name, config, context_reset, close_figs, code_includes,
    )

    # Dark pass — overlay dark rcParams on top of plot_rcparams and render to
    # a parallel filename ({output_base}-dark.png).
    dark_overrides = getattr(config, "plot_rcparams_dark", DARK_RCPARAMS_DEFAULT)
    saved = dict(config.plot_rcparams)
    try:
        config.plot_rcparams = {**saved, **dark_overrides}
        _original_render_figures(
            code, code_path, output_dir, output_base + "-dark", context,
            function_name, config, context_reset, close_figs, code_includes,
        )
    finally:
        config.plot_rcparams = saved

    return result


def _set_plot_template(app, config):
    if not config.plot_template:
        config.plot_template = DUAL_TEMPLATE


def setup(app):
    mpd.render_figures = _render_figures_dual
    app.add_config_value("plot_rcparams_dark", DARK_RCPARAMS_DEFAULT, "html")
    app.connect("config-inited", _set_plot_template)
    return {"version": "0.1", "parallel_read_safe": True}
