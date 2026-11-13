"""Scale every text element of a finished figure so that, printed at the journal's
double-column width (7 in), no label is smaller than the 4.5 pt Analytical Chemistry asks for.

Call scale_fonts(fig, k) immediately before fig.savefig(). It draws the figure once so tick
labels exist, multiplies every visible text's size by k, and returns the smallest size the
text will have at 7 in, which the calling script prints.
"""
import matplotlib.text

FINAL_WIDTH_IN = 7.0


def scale_fonts(fig, k):
    fig.canvas.draw()
    w = fig.get_size_inches()[0]
    smallest = None
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        t.set_fontsize(t.get_fontsize() * k)
        pt = t.get_fontsize() * FINAL_WIDTH_IN / w
        smallest = pt if smallest is None else min(smallest, pt)
    return smallest
