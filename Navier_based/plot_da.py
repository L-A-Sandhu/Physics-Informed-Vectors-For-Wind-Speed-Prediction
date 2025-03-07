import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

# Create fractal-like terrain with multiple Gaussian components
np.random.seed(42)
x = np.linspace(-8, 8, 300)
y = np.linspace(-8, 8, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Add multiple Gaussian components with different parameters
components = [
    {'mu': [-4, -3], 'sigma': [[1.2, 0.9], [0.9, 1.2]], 'weight': 0.3},
    {'mu': [3, 4], 'sigma': [[2.0, -1.2], [-1.2, 1.8]], 'weight': 0.4},
    {'mu': [0, 0], 'sigma': [[0.8, 0], [0, 3.5]], 'weight': 0.2},
    {'mu': [-2, 5], 'sigma': [[1.5, 0.4], [0.4, 0.7]], 'weight': 0.25},
    {'mu': [5, -2], 'sigma': [[0.5, -0.6], [-0.6, 1.2]], 'weight': 0.35}
]

for comp in components:
    rv = multivariate_normal(comp['mu'], comp['sigma'])
    Z += comp['weight'] * rv.pdf(np.dstack([X, Y]))

# Add sinusoidal modulation for complex topography
Z += 0.15 * (np.sin(2*X) * np.cos(3*Y) + 0.5*np.sin(5*X) * np.cos(2*Y))

# Create wind vector field (U, V components)
U = 0.5 * (-Y * np.cos(0.5*X) + 0.3 * np.random.randn(*X.shape))
V = 0.5 * (X * np.sin(0.5*Y)) + 0.3 * np.random.randn(*Y.shape)
speed = np.sqrt(U**2 + V**2)

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot turbulent surface with elevation coloring
surf = ax.plot_surface(X, Y, Z, cmap='plasma',
                      rstride=2, cstride=2,
                      facecolors=cm.viridis(Z/np.max(Z)),
                      alpha=0.85, linewidth=0.4,
                      antialiased=True)

# Add wind vector field arrows
ax.quiver(X[::15, ::15], Y[::15, ::15], Z[::15, ::15]+0.05,
          U[::15, ::15], V[::15, ::15], np.zeros_like(U[::15, ::15]),
          length=0.8, color='white', alpha=0.4,
          arrow_length_ratio=0.3, linewidth=0.7)

# Add contour projections
ax.contourf(X, Y, Z, zdir='z', offset=Z.min()-0.1, cmap='viridis', alpha=0.2)
ax.contour(X, Y, Z, zdir='x', offset=-9, colors='#2F4F4F', linewidths=0.5)
ax.contour(X, Y, Z, zdir='y', offset=9, colors='#2F4F4F', linewidths=0.5)

# Add dynamic annotations with wind-related text
annotation_params = [
    {'pos': (-5, -2, 0.3), 'text': 'Cyclonic Vortex\nVmax ≈ 45 m/s', 'color': '#FF4500', 'rotation': -15},
    {'pos': (4, 3, 0.25), 'text': 'High Pressure Ridge\nΔP = 8 hPa', 'color': '#00BFFF', 'rotation': 25},
    {'pos': (0, 0, 0.4), 'text': 'Turbulence Zone\nε = 0.78', 'color': '#ADFF2F', 'rotation': 0},
    {'pos': (-7, 5, 0.15), 'text': 'Laminar Flow\nRe < 2000', 'color': '#FFFFFF', 'rotation': -45},
    {'pos': (6, -3, 0.2), 'text': 'Wind Shear\n∂u/∂z = 4.2', 'color': '#FFD700', 'rotation': 60},
]

for ann in annotation_params:
    ax.text(*ann['pos'], ann['text'], color=ann['color'],
           rotation=ann['rotation'], ha='center', va='center',
           fontsize=7, bbox=dict(facecolor='black', alpha=0.2, pad=2))

# Add colorbars
cbar1 = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, label='Terrain Elevation')
cbar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=speed.max()), cmap='cool'),
                     ax=ax, shrink=0.3, aspect=10, label='Wind Speed (m/s)')

# Style adjustments
ax.set_xlabel('Longitudinal Coordinate [km]', labelpad=12)
ax.set_ylabel('Latitudinal Coordinate [km]', labelpad=12)
ax.set_zlabel('Atmospheric Potential Energy [J/kg]', labelpad=12)

ax.view_init(elev=38, azim=-50)
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.set_zlim(Z.min()-0.1, Z.max()+0.1)

# Add scientific-looking box
ax.xaxis.pane.set_edgecolor('#404040')
ax.yaxis.pane.set_edgecolor('#404040')
ax.zaxis.pane.set_edgecolor('#404040')
ax.grid(True, linestyle=':', color='grey', alpha=0.4)

# Add title with pseudo-equation
title_text = ("Atmospheric Boundary Layer Turbulence Modeling\n"
              "∂u/∂t + u⋅∇u = -∇p + ν∇²u + F\n"
              "where F ∼ N(0,σ²) represents stochastic wind forcing")
plt.title(title_text, y=1.02, fontsize=10)

plt.tight_layout()
plt.show()