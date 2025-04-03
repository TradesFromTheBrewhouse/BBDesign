import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import plotly.graph_objects as go
from itertools import product

# === Factor ranges ===
temp_range = [35, 37, 39]
co2_range = [4, 5, 6]
seeding_range = [2e4, 5e4, 8e4]

# === Decode helper ===
def decode(val, low, high):
    return ((val + 1) / 2) * (high - low) + low

# === Manually build Box-Behnken design (3 factors) ===
factor_levels = [-1, 0, 1]
bbd_core = [
    [a, b, 0] for a, b in product(factor_levels[::2], factor_levels[::2])
] + [
    [a, 0, b] for a, b in product(factor_levels[::2], factor_levels[::2])
] + [
    [0, a, b] for a, b in product(factor_levels[::2], factor_levels[::2])
]

# Add center points
center_points = [[0, 0, 0]] * 3  # 3 center replicates

# Combine all runs
design = pd.DataFrame(bbd_core + center_points, columns=['A', 'B', 'C'])

# Decode to real values
design['Temp'] = design['A'].apply(lambda x: decode(x, temp_range[0], temp_range[2]))
design['CO2'] = design['B'].apply(lambda x: decode(x, co2_range[0], co2_range[2]))
design['Seeding'] = design['C'].apply(lambda x: decode(x, seeding_range[0], seeding_range[2]))

# Simulate Growth
def simulate_growth(row):
    A, B, C = row['Temp'], row['CO2'], row['Seeding']
    return (
        -0.5 * (A - 37)**2
        - 0.8 * (B - 5)**2
        - 1e-10 * (C - 5e4)**2
        + 100
        + np.random.normal(0, 1)
    )

design['Growth'] = design.apply(simulate_growth, axis=1)

# Fit second-order RSM model
model = ols('Growth ~ A + B + C + A:B + A:C + B:C + I(A**2) + I(B**2) + I(C**2)', data=design).fit()

# Build prediction surface: Temp vs CO2, Seeding fixed
A_vals = np.linspace(-1, 1, 50)
B_vals = np.linspace(-1, 1, 50)
A_grid, B_grid = np.meshgrid(A_vals, B_vals)
C_fixed = 0  # coded center for seeding

grid_df = pd.DataFrame({
    'A': A_grid.ravel(),
    'B': B_grid.ravel(),
    'C': C_fixed,
})
grid_df['A:B'] = grid_df['A'] * grid_df['B']
grid_df['A:C'] = grid_df['A'] * grid_df['C']
grid_df['B:C'] = grid_df['B'] * grid_df['C']
grid_df['I(A ** 2)'] = grid_df['A'] ** 2
grid_df['I(B ** 2)'] = grid_df['B'] ** 2
grid_df['I(C ** 2)'] = grid_df['C'] ** 2

Z = model.predict(grid_df).values.reshape(A_grid.shape)
A_decoded = decode(A_grid, temp_range[0], temp_range[2])
B_decoded = decode(B_grid, co2_range[0], co2_range[2])

# === Streamlit Interface ===
st.title("📊 Simulated Cell Growth (Manual Box-Behnken Design)")
st.markdown("Response surface model based on temperature, CO₂, and seeding density.")

st.subheader("🧪 Experimental Design Data")
st.dataframe(design[['Temp', 'CO2', 'Seeding', 'Growth']])

st.subheader("🌀 Response Surface: Temp vs CO₂ (Seeding fixed)")
fig = go.Figure(data=[go.Surface(z=Z, x=A_decoded, y=B_decoded)])
fig.update_layout(
    title="Predicted Cell Growth",
    scene=dict(
        xaxis_title='Temperature (°C)',
        yaxis_title='CO₂ (%)',
        zaxis_title='Growth'
    )
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📘 Model Summary"):
    st.text(model.summary())
