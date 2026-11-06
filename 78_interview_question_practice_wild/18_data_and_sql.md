# Data and SQL practice

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important for data-heavy MLE roles: clarify grain, time and tie rules.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f171"></a>

See the [SQL lab](SQL_LAB.md) for executable queries, schemas and example results.

## F171. Find the top three vendors by net charge over the last two years.

**Start here.** Aggregate at the intended vendor grain before ranking, with refunds and time boundaries defined.

**Work it through.**

1. Filter transactions using a reproducible as-of timestamp and a half-open date interval.
2. Normalize charge/refund signs or subtract refund amounts according to the schema, then group by vendor id.
3. Sort totals descending with an explicit tie rule; use ROW_NUMBER for exactly three rows or DENSE_RANK when tied ranks should be included.

**Small example.** A vendor with charges 100 and refunds 30 has net charge 70, not gross charge 100.

**Common trap.** Joining a one-to-many vendor table before aggregation and multiplying transaction totals.

**Follow-up.** What if multiple currencies exist? Convert using an explicit rate/date policy before adding amounts.

---

<a id="f172"></a>

## F172. Find each user's consecutive login-day streaks.

**Start here.** Convert repeated events into distinct user-days before detecting consecutive dates.

**Work it through.**

1. Choose timezone and deduplicate (user_id,login_date).
2. Order each user's dates and assign ROW_NUMBER; consecutive runs share a key like login_date minus row_number days.
3. Group by user and run key, then compute length and boundaries with deterministic handling of ties.

**Small example.** Days 1,2,4,5,6 contain streaks of lengths 2 and 3; multiple logins on day 5 do not lengthen either streak.

**Common trap.** Applying row numbers to raw events, so duplicates shift the sequence.

**Follow-up.** How do you count currently active streaks? Restrict run end dates using a stated as-of and grace policy.

---

<a id="f173"></a>

## F173. Find the most frequent event type and its most popular plan.

**Start here.** Solve the two ranking stages at their own aggregation grain.

**Work it through.**

1. Count events by event type over the specified population and period, resolving ties explicitly.
2. Restrict to the chosen event type or types, then count by plan type.
3. Rank plans with a documented row or tie-preserving rule and handle null/unknown plans consistently.

**Small example.** The globally most popular plan need not be the most popular plan inside the most frequent event type.

**Common trap.** Ranking plans globally and joining afterward as if conditional and marginal popularity were the same.

**Follow-up.** What ambiguity should you clarify? Whether “most frequent” counts events, distinct users or distinct sessions.

---

<a id="f174"></a>

## F174. Calculate project duration and budget per employee.

**Start here.** Prevent assignment joins from duplicating project-level amounts before computing employee summaries.

**Work it through.**

1. Define duration units, inclusive/exclusive endpoints and open-project treatment.
2. Aggregate assignment allocations under a documented rule: equal share, hours or explicit percentages.
3. Then summarize by employee, preserving project ids to audit double counts and overlapping periods.

**Small example.** A 1000-unit project assigned equally to four employees contributes 250 units each, not 1000 each, if allocation rather than participation totals is requested.

**Common trap.** Adding full project budgets once per employee without stating that duplication is intentional.

**Follow-up.** Why can summed durations be misleading? Employees may work on overlapping projects simultaneously.

---

<a id="f175"></a>

## F175. Compute cumulative time after deduplicating overlapping intervals.

**Start here.** Union intervals before summing if overlapping activity must count only once.

**Work it through.**

1. Sort intervals by entity and start time after validating endpoints.
2. Merge when the next start is within the current covered interval under the chosen touching-interval convention.
3. Sum merged lengths, using half-open intervals for clear boundary behavior; apply window clipping if the query has a time range.

**Small example.** Intervals [1,4) and [3,6) cover 5 time units together, not 6.

**Common trap.** Subtracting overlaps pairwise when three or more intervals overlap, which can double-correct coverage.

**Follow-up.** What is the complexity? O(N log N) sorting and O(N) merging for each entity's intervals.

---

<a id="f176"></a>

## F176. Build point-in-time training joins.

**Start here.** Historical training rows must contain only information available at their prediction timestamp.

**Work it through.**

1. Define prediction time and feature availability time, not just the original event time.
2. Join each row to the latest feature version whose availability time is no later than prediction time.
3. Test late arrivals, corrections and backfills; store enough history to reproduce what the model could actually have seen.

**Small example.** A transaction recorded at 10:00 but ingested at 10:10 is not available for a prediction at 10:05.

**Common trap.** Using an event timestamp alone when the system learns about events later.

**Follow-up.** Why preserve feature versions? A later correction must not silently rewrite a historical experiment.

---

<a id="f177"></a>

## F177. Design an ETL job that can safely rerun.

**Start here.** A rerun should converge to the intended data state rather than duplicate outputs.

**Work it through.**

1. Define source partitions, stable row keys, transformations and a validation contract.
2. Write to staging, validate row counts and constraints, then publish atomically or with a controlled merge/upsert.
3. Track checkpoints and source watermarks; define late-data reprocessing and schema-evolution policies.

**Small example.** Rerunning yesterday's partition should replace or reconcile that partition, not append it a second time.

**Common trap.** Marking the job complete before its output is durably visible.

**Follow-up.** What should a failure record contain? The input/version boundary and enough diagnostic context to retry or repair without guessing.

---

<a id="f178"></a>

## F178. Optimize a latest-jobs query over a large table.

**Start here.** Match the access path to the actual filters and ordering, then verify using a query plan.

**Work it through.**

1. Inspect selectivity, join cardinalities, sort costs and how many rows must be scanned.
2. Consider an index beginning with selective equality filters followed by the requested order/time field, depending on the database and query.
3. Use keyset pagination when appropriate and measure write/storage overhead as well as read gains.

**Small example.** An ORDER BY created_at DESC LIMIT 20 query may benefit from an aligned index; a function on created_at can prevent a simple index lookup.

**Common trap.** Adding many indexes without checking whether the optimizer uses them or whether writes become too costly.

**Follow-up.** Why can OFFSET pagination get slower? The database may still traverse and discard the skipped rows.

---

<a id="f179"></a>

## F179. Define label windows for click-through-rate analysis.

**Start here.** The numerator and denominator must refer to the same eligible exposures and mature observation windows.

**Work it through.**

1. Define an impression, a valid click, attribution rules, deduplication and the outcome window.
2. Join outcomes to exposures without multiplying rows, then aggregate clicks and impressions by the intended category.
3. Report counts and uncertainty; distinguish unobserved outcomes from confirmed non-clicks when logs are incomplete.

**Small example.** A category with 2 clicks from 2 impressions has CTR 100% but far more uncertainty than one with 10000 clicks from 20000 impressions.

**Common trap.** Averaging per-user CTRs when the requested metric is total clicks divided by total impressions.

**Follow-up.** Why can category comparisons be confounded? Users, positions and exposure policies may differ systematically.

---

<a id="f180"></a>

## F180. Detect a data-pipeline issue before retraining a model.

**Start here.** Operational data checks can explain metric changes more directly than model modifications.

**Work it through.**

1. Compare feature freshness, null rates, value ranges, schema versions and entity-join coverage with normal baselines.
2. Trace a few affected examples end to end from source event to served feature and prediction.
3. Repair and backfill safely, then reassess model performance on mature labels before deciding whether retraining is needed.

**Small example.** An online feature defaulting to zero after a failed join can hurt predictions even though the model artifact is unchanged.

**Common trap.** Retraining immediately on corrupted features and thereby encoding the pipeline defect into a new model.

**Follow-up.** Which alerts need ownership? Every actionable check should identify the responsible pipeline and a response playbook.

---
