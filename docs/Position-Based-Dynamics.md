# Position Based Dynamics

> Matthias Müller, Bruno Heidelberger, Marcus Hennix, John Ratcliff.
> *3rd Workshop in Virtual Reality Interactions and Physical Simulation (VRIPHYS) 2006.* AGEIA.
> PDF: [`posBasedDyn.pdf`](./posBasedDyn.pdf)

## TL;DR

Skip the force layer. Don't accumulate forces → integrate to velocities → integrate to positions; instead
**predict positions** with explicit Euler, then **project** them onto the constraint manifold by repeatedly
solving each constraint `Cⱼ(p) = 0` in Gauss–Seidel fashion. Velocities are recovered post-hoc from how far
the projection moved each particle. Result: **unconditionally stable**, trivially controllable (vertices
can be pinned/teleported mid-step), and collision response is just an inequality constraint generated per
frame. Material stiffness is an artifact of solver-iteration count and timestep — not real stiffness — which
[XPBD](#relationship-to-follow-ups) later fixes.

## The problem

- **Force-based** (FEM, mass-spring with implicit Euler) is physically faithful but expensive and awkward
  for direct manipulation: pinning a vertex or resolving a penetration means injecting forces and hoping the
  integrator follows.
- **Explicit integration** of stiff systems blows up; **implicit** is unconditionally stable but solves a
  large nonlinear system per step.
- Games and real-time apps want stability, controllability, and speed — **visual plausibility, not physical
  accuracy**.

PBD targets that gap.

## Method

### 1. The simulation loop

`N` particles with mass `mᵢ`, inverse mass `wᵢ = 1/mᵢ`, position `xᵢ`, velocity `vᵢ`, plus `M` constraints
`Cⱼ : ℝ^{3nⱼ} → ℝ` of equality (`Cⱼ = 0`) or inequality (`Cⱼ ≥ 0`) type with stiffness `kⱼ ∈ [0,1]`.
Each timestep `Δt`:

```
(5)  ∀i:  vᵢ ← vᵢ + Δt · wᵢ · f_ext(xᵢ)        # external forces → velocity
(6)  dampVelocities(v)                          # global damping (§3.5)
(7)  ∀i:  pᵢ ← xᵢ + Δt · vᵢ                     # PREDICT next position
(8)  generateCollisionConstraints(xᵢ → pᵢ)     # M_coll per-step contacts

(9)  repeat solverIterations times:             # Gauss–Seidel projection
(10)     ∀j ∈ {1..M+M_coll}:  projectConstraint(Cⱼ, p)
(11) end

(13) ∀i:  vᵢ ← (pᵢ − xᵢ) / Δt                  # velocity from positional change
(14) ∀i:  xᵢ ← pᵢ                               # commit
(16) velocityUpdate(v)                          # friction / restitution
```

The key lines are 7, 9–11, 13–14. Line 7 is plain explicit Euler — but only as a *prediction*; it is never
trusted. Lines 9–11 drag the prediction onto the constraint manifold. Line 13 is the Verlet trick in
disguise: if a constraint pushed a particle back, its velocity is killed automatically — that's why
penetrations stop being a problem.

The scheme is **unconditionally stable**: step 14 never extrapolates blindly, it commits to a configuration
the solver already validated. The number of solver iterations is the only stiffness knob: **more iterations
→ stiffer material**, smoothly trending from explicit (1 iter) to implicit-like behavior.

### 2. Constraint projection — the only real math

Given `C(p₁,…,pₙ)`, find `Δp` such that `C(p + Δp) = 0` while conserving linear and angular momentum.

Linear momentum is conserved if `Σᵢ mᵢ Δpᵢ = 0`; angular if `Σᵢ rᵢ × mᵢ Δpᵢ = 0`. Two facts make this clean:

1. For *internal* constraints, `C` is invariant under rigid-body motion (translate/rotate all points → same
   value). Therefore `∇C` is **perpendicular to rigid-body modes** — moving along it can't translate or
   rotate the whole system.
2. So restrict `Δp = λ ∇C(p)`. Linearize: `C(p + Δp) ≈ C(p) + ∇C·Δp = 0`, solve for `λ`.

With per-particle inverse masses, the result (eq. 9):

```
Δpᵢ = −s · wᵢ · ∇_{pᵢ} C(p)

         C(p₁,…,pₙ)
s  =  ─────────────────────────
       Σⱼ wⱼ · |∇_{pⱼ} C(p)|²
```

That's the entire engine. Every constraint type — distance, bending, volume, contact — is just *give me `C`
and `∇C`*.

`wᵢ = 0` ⇒ `Δpᵢ = 0`: **pinning is free**, and the rest of the constraint redistributes correctly around
the fixed point.

### 3. Worked example — distance constraint

`C(p₁, p₂) = |p₁ − p₂| − d`, gradients `∇_{p₁}C = n`, `∇_{p₂}C = −n` with `n = (p₁−p₂)/|p₁−p₂|`. Plug in:

```
              w₁
Δp₁ = − ───────────  (|p₁ − p₂| − d) · n
           w₁ + w₂

              w₂
Δp₂ = + ───────────  (|p₁ − p₂| − d) · n
           w₁ + w₂
```

This is exactly Jakobsen's Verlet rope formula [Jak01]. PBD generalizes it to arbitrary `C`.

### 4. The Gauss–Seidel solver

Constraints are projected **one at a time, sequentially**, each projection seeing the previous ones'
updates. Non-linear Gauss–Seidel.

- Faster convergence than Jacobi (information propagates within a step).
- **Order-dependent**: keep the order constant across frames or oscillation appears in over-constrained
  configurations.
- Pressure waves travel ~one constraint per iteration, so iteration count caps how stiff long chains can
  feel — the canonical PBD weakness.

### 5. Stiffness `k`

Multiply `Δp` by `k ∈ [0,1]`. After `nₛ` iterations the residual is `Δp · (1−k)^{nₛ}` — non-linear in `nₛ`.
Linearize by using

```
k' = 1 − (1 − k)^{1/nₛ}
```

so the residual becomes `Δp · (1−k)`, independent of iteration count. Material stiffness still depends on
`Δt` (this is the central PBD flaw, fixed by [XPBD](#relationship-to-follow-ups)).

### 6. Collisions (§3.4)

Generated per step from the predicted ray `xᵢ → pᵢ`:

- **Continuous**: find entry point `qc`, normal `nc`. Add inequality `C(p) = (p − qc) · nc ≥ 0`, `k = 1`.
- **Static fallback** if you started inside: surface point `qs`, normal `ns`, same form.
- **Two-way (vertex vs. moving triangle)**:
  ```
  C(q, p₁, p₂, p₃) = ±(q − p₁) · [(p₂ − p₁) × (p₃ − p₁)]
  ```
  Invariant under rigid motion ⇒ momentum-preserving by construction.
- **Friction & restitution** are post-hoc velocity edits in line (16): damp tangentially, reflect normally.

Collision constraints are generated **outside** the solver loop (once per step), not regenerated per
iteration. Slight artifacts in pathological cases, big speedup.

### 7. Damping (§3.5)

A clever scheme that damps internal motion only:

```
x_cm = (Σᵢ xᵢmᵢ) / (Σᵢ mᵢ)
v_cm = (Σᵢ vᵢmᵢ) / (Σᵢ mᵢ)
L    = Σᵢ rᵢ × (mᵢ vᵢ),       rᵢ = xᵢ − x_cm
I    = Σᵢ r̃ᵢ r̃ᵢᵀ mᵢ
ω    = I⁻¹ L

∀i:  Δvᵢ = v_cm + ω × rᵢ − vᵢ
∀i:  vᵢ ← vᵢ + k_damping · Δvᵢ
```

`k_damping = 1` ⇒ rigid-body limit; bulk translation/rotation untouched, internal jitter killed.

## Application: cloth (§4)

Three constraint types over a manifold triangle mesh:

| Constraint | `C(...)` | Notes |
|---|---|---|
| Stretch (per edge) | `\|p₁ − p₂\| − l₀` | Spring-free distance preservation |
| Bend (per shared-edge pair) | `arccos(n₁ · n₂) − φ₀` | Dihedral angle, **independent of edge length** |
| Volume (closed mesh) | `Σ_i (p_{t1} × p_{t2}) · p_{t3} − k_p · V₀` | Inflated balloons / characters |

The dihedral-angle bending term is the paper's notable secondary contribution: because it doesn't involve
edge lengths, you can specify low stretch stiffness + high bend stiffness independently, which mass-spring
or [GHDS03] bending struggle with. Gradient derivation in Appendix A.

Self-collision uses spatial hashing [THM*03]; vertex-vs-triangle constraint with cloth thickness `h`:

```
                      (p₂ − p₁) × (p₃ − p₁)
C(q, p₁, p₂, p₃) = (q − p₁) · ─────────────────────  − h
                      |(p₂ − p₁) × (p₃ − p₁)|
```

(Sign flipped if vertex enters from below.) Internal ⇒ momentum-preserving.

**Tearing** is a one-line edit: when an edge stretch exceeds threshold, split a vertex through a plane
perpendicular to the edge.

## Results

PC Pentium 4, 3 GHz; integrated into the *Rocket* viewer.

| Scene | verts | tris | fps |
|---|---|---|---|
| Cloth stripes (Fig. 8), 6 rigid bodies + 3 cloth pieces | — | — | 380+ |
| Self-collision sheet (Fig. 6) | 1364 | 2562 | 30 |
| Tearing cloth (Fig. 10) | 4264 | 8262 | 47 |

## Limitations

- **Material stiffness is timestep- and iteration-dependent.** The fundamental flaw — there is no real
  energy, just a constraint manifold and a relaxation rate. Cloth feels stiffer at smaller `Δt` or higher
  `solverIterations`. Fixed by XPBD.
- **Iteration count caps signal speed**: pressure waves propagate one constraint per iteration, so long
  chains under tension converge slowly.
- **Order-dependent oscillations** in over-constrained scenes if constraint order varies between frames.
- Collisions generated outside the solver loop ⇒ rare missed collisions in pathological motion.

## Relationship to follow-ups

- **XPBD** (Macklin, Müller, Chentanez 2016) gives every constraint a compliance `α = 1/(stiffness · Δt²)`
  and an associated Lagrange multiplier; iterations now converge to a real implicit-Euler solution as
  `α → 0`, removing PBD's timestep dependence. This is the bridge to physically-faithful PBD.
- **Small Steps PBD** (Macklin et al. 2019) exploits substepping in place of iteration count for stiffness.
- Compared with [Rig-Space Physics (2012)](./Rig-Space-Physics.md), which casts implicit Euler as
  minimization of an incremental potential `H = inertia + W(x)`: PBD minimizes nothing, it just projects
  positions onto `C = 0`. Stiffness in PBD is solver-driven; in rig-space physics it is genuine material
  stiffness from `∂²W`. XPBD is the unification — Lagrangian compliance reconciles the two views.
