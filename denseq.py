import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Use LaTeX for text rendering
plt.rcParams['text.usetex'] = True

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define x range
x = np.linspace(-2, 6, 1000)

# Sequence: N(2 - 1/n, 1 + 1/n) converging to N(2, 1)
# Both mean and variance change as n increases

n_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100]
colors = plt.cm.viridis_r(np.linspace(0, 1, len(n_values)))

for i, n in enumerate(n_values):
    mean = 2 - 1/n  # Mean approaches 2 as n → ∞
    variance = 1 + 1/n  # Variance approaches 1 from above as n → ∞
    y = stats.norm.pdf(x, loc=mean, scale=np.sqrt(variance))
    
    # Make later curves more prominent
    alpha = 0.5 + 0.5 * (i / len(n_values))
    linewidth = 2 + 1.5 * (i / len(n_values))
    
    # Make the last density black
    if i == len(n_values) - 1:
        color = 'black'
    else:
        color = colors[i]
    
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_ylim(0, 0.5)
ax.set_xlim(-2, 6)

# Annotate the limiting distribution with arrow
ax.annotate(r'limiting density', xy=(2, 0.4), xytext=(3.5, 0.45),
            fontsize=16, ha='center',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='black'))

plt.tight_layout()
plt.savefig('denseq.pdf')
plt.show()
