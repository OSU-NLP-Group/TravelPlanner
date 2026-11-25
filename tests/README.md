# Evaluator regression tests

Run from the repository root:

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests exercise the real evaluator functions against small in-memory pandas
fixtures. They replace database constructors and the city-state file during
import, restore the working directory, and stub the unused Gradio UI import.
No benchmark download, API key, network call, or model execution is needed.
These are regression tests, not a full validation/test-set benchmark run.

Coverage includes attraction lists with optional trailing separators; exact-first
resource matching (with unique partial-name compatibility); consecutive minimum
night stays; per-person meals/flights and room/vehicle capacity rounding; missing,
ambiguous, and invalid prices; flight route/date matching; and continuous closed
routes with the requested origin and destination. A complete valid itinerary also
exercises both public evaluation entry points.

Unresolvable costs return infinity so the existing budget comparison fails instead
of treating the item as free. Identical duplicate database rows are deduplicated;
conflicting matches fail lookup rather than depending on row order. Existing
missing-information checks remain responsible for omitted plan components.
