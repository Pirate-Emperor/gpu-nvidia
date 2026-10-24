# Documentation

Start with the [README](../README.md), then use this list for details.

| Read this | For |
|---|---|
| [Project two-pager](PROJECT_TWO_PAGER.md) | Problem, kernel and current status |
| [Course checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md) | Required assignment items |
| [TODO checklist](TODO_CHECKLIST.md) | Next work in order |
| [Kernel walkthrough](CUTILE_KERNEL_WALKTHROUGH.md) | What the cuTile code does |
| [Mentor walkthrough](MENTOR_STEP_BY_STEP_WALKTHROUGH.md) | How to run the tests and demo |
| [Testing contract](TESTING_CONTRACT.md) | Inputs, tolerance and regression rules |
| [Test matrix](TEST_MATRIX.md) | Test coverage and saved outcomes |
| [Baseline ladder](BASELINE_LADDER.md) | What each comparison tells us |
| [Environment notes](ENVIRONMENT.md) | GPU, driver and package limits |
| [NCU analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) | Existing profiler results |
| [20 October results](CUTILE_RTX4060_RESULTS_2026-10-20.md) | Timing and accuracy tables |
| [Experiment report](EXPERIMENT_REPORT.md) | Decisions, failures and prior tests |
| [Execution evidence](EXECUTION_EVIDENCE_INDEX.md) | Workflow runs and artifacts |
| [Full report](../Report.md) | Detailed experiment history |

## Current state

The last complete native GPU run was on commit bf5a623. The cuTile kernel was compacted afterward. Rerun the current source before treating the older timing and profiler results as current. The framework attention comparison, clean-cache JIT timing and another profile-driven improvement remain open.
