# Benchmark methodology

Every published benchmark should include the following.

## System

- CPU model, physical/logical core count, memory, NUMA layout;
- accelerator model, memory, driver, and framework versions;
- operating system and power/performance settings;
- release build confirmation from `rf.core_build_profile()`;
- reinfors and comparison-library commits.

## Workload

- resolved engine configuration and fingerprint;
- game, observation encoder, reward, policy, learner, chance mode, and seeds;
- model architecture, parameter count, precision, device, and compilation settings;
- parallel games, search budget, cache capacity, stream depth, and thread settings;
- warm-up, measured duration, repeats, and uncertainty calculation.

## Measurements

Report both throughput and the work that produced it: records, decisions, episodes,
inference calls and rows, callback time, effective callback batch size, cache behavior, search
expansions, and episode lengths. For concurrent runs, separately report collection, training,
idle time, queue occupancy, and policy lag.

Comparisons must align rules, observation semantics, chance behavior, search budget, and
model outputs. If they cannot align, label the difference rather than presenting the result
as like-for-like.

## Publication checklist

- [ ] Commands and configs committed
- [ ] Raw machine-readable results available
- [ ] Multiple runs with uncertainty
- [ ] Correctness checks pass before timing
- [ ] No debug builds
- [ ] Warm-up excluded consistently
- [ ] Charts generated from checked-in analysis
- [ ] Limitations and non-equivalent semantics disclosed
