pub_st = dict()
pub_st['figure.figsize'] = (10, 9)
pub_st['axes.titlesize'] = 18 # 28
pub_st['axes.labelsize'] = 20 # 24
pub_st['legend.fontsize'] = 14 # 18
pub_st['xtick.labelsize'] = 18 # 22
pub_st['ytick.labelsize'] = 18 # 22

# Set x axis
pub_st['xtick.direction'] = 'in'
pub_st['xtick.major.size'] = 5
pub_st['xtick.major.width'] = 1.5
pub_st['xtick.minor.size'] = 3
pub_st['xtick.minor.width'] = 1
pub_st['xtick.minor.visible'] = True
pub_st['xtick.top'] = True

# Set y axis
pub_st['ytick.direction'] = 'in'
pub_st['ytick.major.size'] = 5
pub_st['ytick.major.width'] = 1.5
pub_st['ytick.minor.size'] = 3
pub_st['ytick.minor.width'] = 1
pub_st['ytick.minor.visible'] = True
pub_st['ytick.right'] = True

# Set line widths
pub_st['axes.linewidth'] = 1.5
pub_st['grid.linewidth'] = 0.5
pub_st['lines.linewidth'] = 1.5
pub_st["axes.axisbelow"] = False

# Remove legend frame
pub_st['legend.frameon'] = False

# Always save as 'tight'
pub_st['savefig.bbox'] = 'tight'
pub_st['savefig.pad_inches'] = 0.05

# Font Family
pub_st['font.family'] = 'serif'
pub_st['font.serif'] = ['cmr10', 'Computer Modern Serif', 'DejaVu Serif']
pub_st['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Avant Garde', 'sans-serif']
pub_st['mathtext.fontset'] = 'dejavusans'
# pub_st['mathtext.fontset'] = 'cm'
pub_st['axes.formatter.use_mathtext'] = True

# Use LaTeX for math formatting
# pub_st['text.usetex'] = True
pub_st['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'