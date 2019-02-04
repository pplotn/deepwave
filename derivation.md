# Deepwave Full-Waveform Inversion

## Acoustic wave equation

### Variable density derivation

From Newton's Second law ($F = ma$):
\begin{equation}
f_i - \nabla P = \rho\partial_{t^2}u_i,
\end{equation}
where $f_i$ is the external force in the $i$ direction, $P$ is the pressure, $\rho$ is the density, and $u_i$ is the displacement in the $i$ direction.

Dividing by $\rho$ and taking the divergence, we obtain
\begin{equation}
\nabla\cdot \frac{f_i}{\rho} - \nabla\cdot\frac{1}{\rho}\nabla P = \nabla\cdot\partial_{t^2}u_i,
\end{equation}

We also note that displacement gradients change density,
\begin{equation}
\Delta \rho = -\rho\nabla\cdot u_i,
\end{equation}
where $\Delta \rho$ is the change in density.

The adiabatic bulk modulus (incompressibility) is defined as:
\begin{equation}
\kappa = \rho\frac{\Delta P}{\Delta\rho} = \rho c^2,
\end{equation}
where $c$ is the compressional wave (P wave) speed.

Combining the previous two relations, we obtain:
\begin{equation}
\rho\frac{\Delta P}{-\rho\nabla\cdot u_i} = \rho c^2,
\end{equation}
or,
\begin{equation}
\Delta P = -\rho c^2 \nabla\cdot u_i.
\end{equation}

Taking the second time derivative,
\begin{equation}
\partial_{t^2} P = -\rho c^2 \nabla\cdot \partial_{t^2} u_i.
\end{equation}

Combining this with an earlier equation,
\begin{equation}
\nabla\cdot \frac{f_i}{\rho} - \nabla\cdot\frac{1}{\rho}\nabla P = -\frac{1}{\rho c^2}\partial_{t^2} P.
\end{equation}

Multiplying by $-\rho c^2$, swapping sides, changing the order of terms, and including the possibility of a pressure source, $s$,
\begin{equation}
\partial_{t^2} P = \rho c^2\nabla\cdot\frac{1}{\rho}\nabla P - \rho c^2\nabla\cdot \frac{f_i}{\rho} + s.
\end{equation}

### Constant density derivation

Taking the previous result and setting $\rho$ to a constant in space,
\begin{equation}
\partial_{t^2} P = c^2\nabla^2 P - c^2\nabla\cdot f_i + s,
\end{equation}
which is the usual scalar wave equation.

where $v = \partial_t u$ is the velocity in the $i$ direction
