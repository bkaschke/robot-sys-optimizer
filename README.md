# robot-sys-optimizer

A browser-based tool for robotic process cells (sanding, scanning, grinding, gripping). It finds **where the robot base or its carrier should stand** and **the shortest end effector that reaches every target** without collision or overload, then exports TCP poses, an optimized tool URDF and an HTML report.

Built in Python: a [Viser](https://viser.studio) GUI, Pinocchio and Coal for kinematics and collision, and OR-Tools and CMA-ES for optimization. It ships with a UR20 model.

What changed since `main`: [CHANGELOG.md](CHANGELOG.md).

**Start here: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)** (setup on a new computer, quickstart, features, examples). Inside the running tool: **User guide → 📖 Open user guide** (the same guide as HTML, `docs/USER_GUIDE.html`).

**Windows, no conda or internet needed:** download the installer (`RobotCellOptimizer-<version>-setup.exe`, per user, no admin rights) or the zip from the Releases page (GitHub or GitLab) and run it; see [section 2.1 of the guide](docs/USER_GUIDE.md#21-download-and-run-windows-installer-or-zip). Your data stays in `%LOCALAPPDATA%\RobotCellOptimizer`. For development, set up from source:

```bash
# after setup (Miniforge env "rso", see the guide)
scripts\run_app.bat          # Windows; or: <rso python> app.py [--dark]
# then open http://localhost:8080
```

UR robot models © 2023 Universal Robots A/S, subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation (`assets/robots/ur20/meshes/LICENSE.txt`). Design plan: `docs/PROJECT_PLAN.md`. Developer notes: `CLAUDE.md`.
