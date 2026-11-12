# Robot Learning Project Selection Task

A mini project for undergraduate robot-learning applicants.

There will be two phases. Phase A is a recorded presentation for this provided task as shown below. Phase B is a more detailed face-to-face interview, introduction to our projects and possible group roles which will follow up if you pass Phase A.

## Setup

No GPU or local simulator is required. Google Colab is the intended interface.

Open `colab/robot_learning_selection.ipynb`, upload this folder, and run the notebook
cells to reproduce the public metrics.

You may use Chinese or English at your preference and may use online resources / AI assistants.

## Submission (Deadline: 10/8)

- Your code (typically `src/policy.py` and any helpers you add)
- Record a video presentation following the instruction of `README_submission.md`
- Any Demo if possible that you think it's helpful for our evaluation

## Task

A point robot moves in a bounded 2D workspace with **random axis-aligned obstacles**
(blocks). You only receive **relative** observations `[Δx, Δy]` toward the goal —
**not** absolute coordinates and **not** a map of the blocks. Actions are 2D velocity
commands. Everything runs on CPU in NumPy.

You are given a **fixed demonstration budget** (`make_dataset`, default 4000 noisy
expert samples) and a **linear behavioral-cloning baseline**. The expert uses
short-horizon path information internally when generating demos; your policy must
learn from `(observation, action)` pairs only. Under that budget, do as well as you
can. There is no single required algorithm: explain what you tried, how you define
“good”, and the trade-offs you see.

Even if you fail, **don't worry**, that happens a lot of time in research. Summary why you fail, what you have learned from this failure and what you try to solve this problem is what we treasure.

Organizers may re-run your policy on held-out scenarios (different layouts, more
blocks, noise, shorter horizons). **Do not hard-code test targets, obstacle layouts,
or scenario lists.**

## Public metrics

The evaluator reports:

- **success rate** — final distance `< 0.05` and not inside a block
- **mean final distance**
- **mean episode length (steps)**
- **collision episode rate** — fraction of rollouts with at least one blocked step

## Environment summary

- Step size `0.08`, default max **80** steps on the public set
- If a step would enter a block, the robot **does not move** (collision counted)
- Demonstrations: grid-based expert with **action noise**
- Observations may include small noise during evaluation

## Pay attention
- we don't answer related tech question such as "there's a bug, I can't fix it", "I don't have time to do this task, Could you give me more time" or "I can't set up the environment."
- When you face a problem, you may search the solution online, ask your favorite AI agent, or try to solve it by your own. If you really can't solve it, post the question in the group chat. We will answer it as quickly if it's really something beyond your ability.
- Try your best! We don't care about your background, we care about your attitude and effort. 
