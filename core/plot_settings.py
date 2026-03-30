
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, mpl

#### colors definitions
snow = '#FFFFF0'
blackblue_mpl = [14/256, 15/256, 31/256, 1.0]
inactive_mpl = [47/256, 52/256, 106/256, 1.0]
inactive_light_mpl = [131/256, 137/256, 201/256, 1.0]
electric_blue_mpl = [0/256, 240/256, 255/256, 1.0]
red_salmon_mpl = [254/256, 69/256, 62/256, 1.0]

#### style scheme choice
background_color = blackblue_mpl
ax_color = snow
image_1 = background_color
image_2 = inactive_mpl
image_3 = inactive_light_mpl
plot_line_1 = electric_blue_mpl
plot_line_2 = red_salmon_mpl

colors_in_cycle = [plot_line_1, plot_line_2]
colors_in_cycle.extend(mpl.rcParams['axes.prop_cycle'].by_key()['color'])
linestyles_in_cycle = ['-' for i in range(len(colors_in_cycle))]
linestyles_in_cycle[1] = ':'

fontsize_mpl = 14

mpl.rcParams['axes.prop_cycle'] = cycler(color=colors_in_cycle) + cycler(linestyle=linestyles_in_cycle)
mpl.rcParams['figure.facecolor'] = background_color
mpl.rcParams['axes.facecolor'] = background_color
mpl.rcParams['axes.labelcolor'] = ax_color
mpl.rcParams['axes.titlecolor'] = inactive_light_mpl
mpl.rcParams['axes.edgecolor'] = ax_color
mpl.rcParams['axes.labelsize'] = fontsize_mpl
mpl.rcParams['xtick.color'] = ax_color
mpl.rcParams['xtick.labelcolor'] = ax_color
mpl.rcParams['xtick.labelsize'] = fontsize_mpl
mpl.rcParams['ytick.color'] = ax_color
mpl.rcParams['ytick.labelcolor'] = ax_color
mpl.rcParams['ytick.labelsize'] = fontsize_mpl
mpl.rcParams['text.color'] = ax_color
mpl.rcParams['grid.color'] = ax_color
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.4
mpl.rcParams['font.size'] = fontsize_mpl
mpl.rcParams['font.stretch'] = 'condensed'
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.edgecolor'] = '0.0'
mpl.rcParams['legend.fontsize'] = fontsize_mpl
mpl.rcParams['legend.loc'] = 'upper right'



cmap_image = LinearSegmentedColormap.from_list(name = 'cmap_image',
                                               colors = [image_1, image_2, image_3],
                                                 N = 256, gamma = 1
                                               )
mpl.colormaps.register(cmap=cmap_image)


black        = [  0/256,   0/256,   0/256, 1.0]
purple       = [ 80/256,   4/256, 143/256, 1.0]
light_purple = [ 80/256,   4/256, 143/256, 0.6]
red          = [187/256,   0/256,   0/256, 1.0]
light_red    = [187/256,   0/256,   0/256, 0.6]
# bright_green = [102/256, 255/256,   0/256, 1.0]


cmap_weights = LinearSegmentedColormap.from_list(name = 'cmap_weights',
                                                 colors = [light_purple, purple,
                                                           black, red, light_red],
                                                 N = 256, gamma = 1
                                                )

#### #register cmap
mpl.colormaps.register(cmap=cmap_weights)

