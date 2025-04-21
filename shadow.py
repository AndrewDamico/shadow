import streamlit as st
import pulp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("Interactive LP Sensitivity and Duality Explorer")
st.markdown("""
This app solves a linear programming (LP) problem and its dual while allowing you to interactively vary parameters for sensitivity analysis.

**Primal Problem:**

Maximize  
\(Z = a \, x_1 + 5\, x_2\)

Subject to:  
\(2x_1 + 3x_2 \leq RHS_1\)  
\(x_1 + x_2 \leq 4\)  
\(x_1, x_2 \geq 0\)

By default, **\(a = 3\)** (the coefficient for \(x_1\)) and **\(RHS_1 = 8\)** for the first constraint.

**Dual Problem (derived from the primal):**

Minimize  
\(W = (RHS_1)\, y_1 + 4\, y_2\)

Subject to:  
\(2y_1 + y_2 \geq a\)  
\(3y_1 + y_2 \geq 5\)  
\(y_1, y_2 \geq 0\)

The app uses PuLP to solve both the primal and dual problems and provides graphical and tabular sensitivity analysis.
""")

##########################################
# Sidebar: Input Parameters
##########################################
st.sidebar.header("Input Parameters")

# Slider for varying the RHS of Constraint 1 (2x₁ + 3x₂ ≤ RHS)
rhs1 = st.sidebar.slider("RHS for Constraint 1 (2x₁ + 3x₂ ≤ RHS)", min_value=8, max_value=12, value=8, step=1)

# Slider for varying the coefficient for x₁ in the objective function
coef_x1 = st.sidebar.slider("Coefficient for x₁ in the objective", min_value=3, max_value=6, value=3, step=1)

# Fixed parameters
rhs2 = 4       # For Constraint: x₁ + x₂ ≤ 4
coef_x2 = 5    # Coefficient for x₂ in the objective

##########################################
# Functions to Solve the LPs
##########################################
def solve_primal(a, rhs1_value):
    """Solve the primal LP with the given coefficient a and RHS value for Constraint 1."""
    # Define the problem: maximize a*x1 + 5*x2
    prob = pulp.LpProblem("Primal_LP", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)

    # Objective function
    prob += a * x1 + coef_x2 * x2, "Objective"

    # Constraints
    prob += 2 * x1 + 3 * x2 <= rhs1_value, "Constraint1"
    prob += x1 + x2 <= rhs2, "Constraint2"

    # Solve the LP
    prob.solve()

    # Record the solution values
    solution = {
        "x1": x1.varValue,
        "x2": x2.varValue,
        "Objective": pulp.value(prob.objective)
    }
    
    # Compute slack for each constraint (slack=0 indicates the constraint is binding)
    slack1 = rhs1_value - (2 * x1.varValue + 3 * x2.varValue)
    slack2 = rhs2 - (x1.varValue + x2.varValue)
    solution["Slack_Constraint1"] = slack1
    solution["Slack_Constraint2"] = slack2

    # Get reduced costs for the variables (if available)
    solution["ReducedCost_x1"] = x1.dj if hasattr(x1, 'dj') else None
    solution["ReducedCost_x2"] = x2.dj if hasattr(x2, 'dj') else None

    # Get dual (shadow price) values for each constraint
    duals = {}
    for name, constraint in prob.constraints.items():
        duals[name] = constraint.pi if hasattr(constraint, 'pi') else None
    solution["Duals"] = duals
    
    # Identify binding constraints (using a tolerance to account for floating point errors)
    binding = [name for name, slack in zip(["Constraint1", "Constraint2"], [slack1, slack2]) if abs(slack) < 1e-5]
    solution["BindingConstraints"] = binding

    return solution, prob

def solve_dual(a, rhs1_value):
    """Solve the dual LP corresponding to the given primal parameters."""
    dual_prob = pulp.LpProblem("Dual_LP", pulp.LpMinimize)
    y1 = pulp.LpVariable("y1", lowBound=0)
    y2 = pulp.LpVariable("y2", lowBound=0)

    # Dual objective: minimize rhs1_value*y1 + 4*y2 (note: rhs2 is fixed at 4)
    dual_prob += rhs1_value * y1 + rhs2 * y2, "DualObjective"

    # Dual constraints based on the primal coefficients
    dual_prob += 2 * y1 + y2 >= a, "DualConstraint1"
    dual_prob += 3 * y1 + y2 >= coef_x2, "DualConstraint2"

    # Solve the dual LP
    dual_prob.solve()

    dual_solution = {
        "y1": y1.varValue,
        "y2": y2.varValue,
        "DualObjective": pulp.value(dual_prob.objective)
    }
    return dual_solution, dual_prob

##########################################
# Solve and Display Primal Results
##########################################
primal_solution, primal_prob = solve_primal(coef_x1, rhs1)
st.header("Primal LP Solution")
st.write(f"**Optimal x₁:** {primal_solution['x1']}")
st.write(f"**Optimal x₂:** {primal_solution['x2']}")
st.write(f"**Optimal Objective Value (Z):** {primal_solution['Objective']:.2f}")
st.write("**Binding Constraints:**", primal_solution["BindingConstraints"])
st.write("**Shadow Prices (Dual Values):**", primal_solution["Duals"])
st.write("**Reduced Costs:**", f"x₁ = {primal_solution['ReducedCost_x1']}, x₂ = {primal_solution['ReducedCost_x2']}")

##########################################
# Graphical Representation of the Feasible Region
##########################################
st.subheader("Feasible Region and Optimal Solution")
fig, ax = plt.subplots()

# Define x values for plotting
x_vals = np.linspace(0, 10, 300)

# Constraint lines:
# For Constraint1: 2x₁ + 3x₂ = rhs1  =>  x₂ = (rhs1 - 2x₁)/3
y_constraint1 = (rhs1 - 2 * x_vals) / 3.0
# For Constraint2: x₁ + x₂ = rhs2  =>  x₂ = rhs2 - x₁
y_constraint2 = rhs2 - x_vals

# Plot the constraint lines
ax.plot(x_vals, y_constraint1, label=f'2x₁ + 3x₂ = {rhs1}')
ax.plot(x_vals, y_constraint2, label=f'x₁ + x₂ = {rhs2}')

# Shade the feasible region (only where constraints are met and y is positive)
y_feasible = np.minimum(y_constraint1, y_constraint2)
ax.fill_between(x_vals, y_feasible, 0, where=(y_feasible > 0), color='gray', alpha=0.3)

# Mark the optimal solution
ax.plot(primal_solution["x1"], primal_solution["x2"], 'ro', markersize=8, label='Optimal Solution')

ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_title("Feasible Region")
ax.legend()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
st.pyplot(fig)

##########################################
# Solve and Display Dual Results
##########################################
dual_solution, dual_prob = solve_dual(coef_x1, rhs1)
st.header("Dual LP Solution")
st.write(f"**Optimal y₁ (shadow price for Constraint 1):** {dual_solution['y1']}")
st.write(f"**Optimal y₂ (shadow price for Constraint 2):** {dual_solution['y2']}")
st.write(f"**Dual Objective Value (W):** {dual_solution['DualObjective']:.2f}")

st.markdown("""
**Comparison:**  
The dual objective value should, in theory, equal the primal objective value when both problems are solved optimally.  
The dual variable values (shadow prices) provide insight into how much the objective function would improve with a marginal increase in the RHS values.
""")
st.write("Primal Objective Value:", f"{primal_solution['Objective']:.2f}")
st.write("Dual Objective Value:", f"{dual_solution['DualObjective']:.2f}")

##########################################
# Sensitivity Analysis
##########################################
st.header("Sensitivity Analysis")

# Part B: RHS Sensitivity Analysis
st.subheader("RHS Sensitivity (Varying RHS of Constraint 1)")
rhs_values = [8, 9, 10, 12]
rhs_results = []
for r in rhs_values:
    sol, _ = solve_primal(coef_x1, r)
    rhs_results.append({
        "RHS Constraint1": r,
        "x₁": sol["x1"],
        "x₂": sol["x2"],
        "Objective": sol["Objective"],
        "Binding Constraints": ", ".join(sol["BindingConstraints"]) if sol["BindingConstraints"] else "None"
    })
df_rhs = pd.DataFrame(rhs_results)
st.dataframe(df_rhs)

st.markdown("""
*This table shows the changes in the optimal solution and objective value as the RHS of Constraint 1 is varied.  
The dual values (shadow prices) indicate the sensitivity of the objective function to changes in the RHS, assuming the basis remains unchanged.*
""")

# Part C: Objective Coefficient Sensitivity for x₁
st.subheader("Objective Coefficient Sensitivity (Varying Coefficient for x₁)")
coef_values = [3, 4, 5, 6]
coef_results = []
for c in coef_values:
    sol, _ = solve_primal(c, rhs1)
    coef_results.append({
        "Coefficient for x₁": c,
        "x₁": sol["x1"],
        "x₂": sol["x2"],
        "Objective": sol["Objective"],
        "Reduced Cost x₁": sol["ReducedCost_x1"]
    })
df_coef = pd.DataFrame(coef_results)
st.dataframe(df_coef)

st.markdown("""
*This table illustrates how the optimal solution, objective value, and the reduced cost for \(x_1\) change as its coefficient is varied.  
A nonzero reduced cost for \(x_1\) suggests that a change in the optimal basis may occur.*
""")
