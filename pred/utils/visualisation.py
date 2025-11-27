from matplotlib.lines import Line2D
from copy import copy
from abc import ABC, abstractmethod
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches

class Style(ABC):
    def __init__(self, color='b', alpha=1):
        self.color = color
        self.alpha = alpha

    def get_style(self):
        return self.__dict__

    def get_legend_element(self, label):
        """Public method to generate the legend handle."""
        return self._create_legend_handle(label)

    @abstractmethod
    def _create_legend_handle(self, label):
        """Abstract method to be implemented by subclasses."""
        pass


class Line(Style):
    def __init__(self, linestyle='-', linewidth=1, **kwargs):
        super().__init__(**kwargs)
        self.linestyle = linestyle
        self.linewidth = linewidth

    def _create_legend_handle(self, label):
        return [Line2D(
            [0], [0], linestyle=self.linestyle, linewidth=self.linewidth,
            color=self.color, alpha=self.alpha, label=label
        )]


class Marker(Style):
    def __init__(self, marker='x', markersize=10, **kwargs):
        super().__init__(**kwargs)
        self.marker = marker
        self.markersize = markersize

    def _create_legend_handle(self, label):
        return [Line2D(
            [0], [0], linestyle='none', marker=self.marker, markersize=self.markersize,
            color=self.color, alpha=self.alpha, label=label
        )]


class Patch(Style):
    def __init__(self, hatch=None, edgecolor=None, **kwargs):
        super().__init__(**kwargs)
        self.hatch = hatch
        self.edgecolor = edgecolor

    def _create_legend_handle(self, label):
        return [mpatches.Patch(
            color=self.color, alpha=self.alpha, hatch=self.hatch,
            edgecolor=self.edgecolor, label=label
        )]


def from_styles_create_legend_elements_v1(styles):
    legend_elements = []
    for key in styles.keys():
        st = copy(styles[key])
        if 'edgecolor' in st:
            st = from_patch_style_create_line_style(st)
        if 'markersize' in st:
            st['markersize'] = 5
        if not 'linestyle' in st:
            st['linestyle'] = 'none'
        line = Line2D([0], [0],
                      label=key,
                      **st)
        legend_elements.append(line)
    return legend_elements


def from_patch_style_create_line_style(patch_style):
    linestyle = {}
    linestyle['linewidth'] = patch_style['linewidth']
    linestyle['color'] = patch_style['edgecolor']
    linestyle['linestyle'] = patch_style['linestyle']
    linestyle['alpha'] = patch_style['alpha']
    return linestyle