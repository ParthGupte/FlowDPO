# ODE Solvers

This folder contains different ODE solver implementations used for solving flow matching ODEs.

---

## ✅ Purpose

Each solver implements a specific numerical integration method (e.g., Euler, RK4)  
that solves the ODE of the form:
  
> \( \frac{dz}{dt} = v(z, t) \)

where `v(z, t)` is provided by a flow model.

---

## 🚀 Adding a New ODE Solver

To create a new solver:

1. Create a new Python file in this folder (e.g., `my_custom_solver.py`).

2. Import the base solver class:
    ```python
    from solver import Solver
    ```

3. Define your solver class by inheriting from `Solver`:
    ```python
    class MyCustomSolver(Solver):
        def __init__(self, flow_model):
            super().__init__()
            self.flow_model = flow_model

        def step(self, z_t, t, dt):
            # Implement your custom ODE step logic here
            pass

        def solve(self, X_0, n_steps):
            # Implement the solve loop calling self.step()
            pass
    ```

4. Follow the same structure as the existing solvers:
    - `__init__(self, flow_model)`  
    - `step(self, z_t, t, dt)` → Computes the next step.
    - `solve(self, X_0, n_steps)` → Integrates from t=0 to t=1 over `n_steps`.
