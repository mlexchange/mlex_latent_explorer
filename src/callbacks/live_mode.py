from dash import Input, Output

from app_layout import app


@app.callback(
    Output("data-selection", "style"),
    Output("dim-red-controls", "style"),
    Output("clustering-controls", "style"),
    Output("heatmap-controls", "style"),
    Input("go-live", "n_clicks"),
)
def toggle_controls(n_clicks):
    # accordionitems share borders, hence we add border-top for heatmap to avoid losing the top
    # border when hiding the top items
    if n_clicks is not None and n_clicks % 2 == 1:
        return [{"display": "none"}] * 3 + [
            {"border-top": "1px solid rgb(223,223,223)"}
        ]
    else:
        return [{"display": "block"}] * 3 + [{"display": "block"}]
