# robot-sys-optimizer

A browser-based tool for robotic process cells (sanding, scanning, grinding, gripping). It finds **where the robot base or its carrier should stand** and **the shortest end effector that reaches every target** without collision or overload, then exports TCP poses, an optimized tool URDF and an HTML report.

Built in Python: a [Viser](https://viser.studio) GUI, Pinocchio and Coal for kinematics and collision, and OR-Tools and CMA-ES for optimization. It ships with a UR20 model.

**Start here: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)** (setup on a new computer, quickstart, features, examples). Inside the running tool: **User guide → 📖 Open user guide** (the same guide as HTML, `docs/USER_GUIDE.html`).

```bash
# after setup (Miniforge env "rso", see the guide)
scripts\run_app.bat          # Windows; or: <rso python> app.py [--dark]
# then open http://localhost:8080
```

UR robot models © 2023 Universal Robots A/S, subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation (`assets/robots/ur20/meshes/LICENSE.txt`). Design plan: `docs/PROJECT_PLAN.md`. Developer notes: `CLAUDE.md`.
