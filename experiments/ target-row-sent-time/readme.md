# Target-All-Row: Entropy Progression Experiments

This folder contains comparison plots of entropy progression over message sent time for different dummy-message strategies.

The strategies compared in each figure are:
- `baseline`
- `client_dummies`
- `link_based_dummies`
- `multiple_hop_dummies`

## Folder Contents

- `entropy_progression_window_0.png`
	- Entropy plotted with **window = 0** interpretation used as **no smoothing** (raw values).
- `entropy_progression_window_5.png`
	- Entropy plotted with a moving-average smoothing window of 5 points.
- `entropy_progression_window_25.png`
	- Entropy plotted with a moving-average smoothing window of 25 points.

## Smoothing Notes

- **Window 0**: no smoothing, raw entropy values are shown.
- **Window 5**: light smoothing, short-term fluctuations are reduced.
- **Window 25**: strong smoothing, trends are emphasized and short-term noise is heavily reduced.

Use window 0 to inspect true point-level behavior, and windows 5/25 to compare broader trend differences between dummy strategies.
