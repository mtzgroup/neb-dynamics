# Annoyed User

This user is impatient, adversarial, and highly sensitive to latency, ambiguity, and weak feedback.

## Core Traits

- Clicks buttons multiple times if the UI does not immediately show a message, animation, disabled state, or other confirmation.
- Assumes silence means the click did not register.
- Treats delays over about 1 second as suspicious and delays over several seconds as unacceptable unless the UI is visibly busy.
- Distinguishes sharply between "the chemistry is taking time" and "the wrapper/UI is wasting time."
- Interprets low-level backend errors as product failures unless they are translated into clear user-facing explanations.
- Re-tests the same action immediately after failure to see whether the product is robust or brittle.
- Reloads or re-initializes from an already loaded workspace to probe whether state transitions are efficient and obvious.

## What This User Tries To Break

- Initialize a new network, then click initialize again before the UI responds.
- Load an existing workspace, then immediately initialize a different network from the same session.
- Click `Apply Reactions` several times in a row on the same node.
- Click node minimization repeatedly, especially after a prior failure or convergence.
- Click NEB queueing repeatedly and quickly after completion or failure.
- Switch between tabs while actions are running to see whether status is globally visible.
- Select nodes and edges while polling is ongoing to expose UI jank or stale detail panels.

## What Annoys This User

- A request is accepted, but there is no visible in-progress confirmation right away.
- The page polls slowly enough that a running action is not reflected for many seconds.
- State refreshes themselves are slow, causing the whole page to feel sticky or blocked.
- Buttons remain clickable with no immediate disable or rejection message.
- Duplicate clicks produce inconsistent responses depending on timing.
- The UI shows optimistic activity, but the real server state lags badly behind it.
- Raw backend exceptions leak through without being translated into a clear explanation and remedy.
- A loaded workspace makes all later interactions sluggish because state reconstruction is too expensive.

## Performance Expectations

- Clicking any action should produce immediate feedback in under 1 second.
- If the backend has accepted work, the UI should visibly enter a running state almost immediately.
- Polling endpoints should stay lightweight enough that ordinary refreshes do not feel like a blocking action.
- Expensive chemistry is acceptable only if the wrapper stays responsive and clearly indicates progress.

## Observed Annoyances From Current Drive Testing

- `POST` actions often return quickly, but the first real `/api/state` update can take about 10 seconds or more on a loaded workspace.
- `Apply Reactions` can show the largest gap between click acceptance and confirmed running state.
- Re-initializing from an already loaded network is accepted fast, but visible progress still depends on a very slow state refresh.
- Template application currently failed with:
  - `AttributeError: 'Molecule' object has no attribute 'substitute_group'`
- Local NEB execution surfaced a raw environment/backend error:
  - `OSError: [Errno 8] Exec format error: 'crest'`

## How To Use This Persona In Future Debugging

- Simulate repeated clicks on every long-running action.
- Measure time from click to accepted response.
- Measure time from accepted response to first visible running state.
- Measure raw `/api/state` latency on both empty and loaded workspaces.
- Check whether duplicate requests are rejected immediately and consistently.
- Check whether failure messages are actionable for a non-developer user.
- Treat any wrapper-induced lag separately from true chemistry runtime.
