from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def embed_figure(parent, figure: Figure, with_toolbar: bool = True) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)

    if with_toolbar:
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

    return canvas


def clear_children(parent) -> None:
    for child in list(parent.winfo_children()):
        child.destroy()
