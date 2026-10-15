# why benchmark

| category  | types      | object                        | baseline                  | focus                          |
| --------- | ---------- | ----------------------------- | ------------------------- | ------------------------------ |
| layer     | LLM        | LLM                           | BECL, HumanEval, MBPP     | code quality, tool calling     |
|           | Agent      | Agent                         | SWE-bench, Agentbench     | e2e problem solve, multi steps |
| method    | auto       | program validation            | Unit, CI pipeline         |                                |
|           | human      | code semanic / quality        | SME                       |                                |
|           | mixed      | auto + human                  | analyze the failed sample |                                |
| demission | function   | problem solved                | testing case pass %       |                                |
|           | quality    | readibility + maintainability | Human analyze, SCA        |                                |
|           | efficiency | time/token/cost               | statistics                |                                |
|           | steability | rebustness and resilence      | multi rounds, boundry     |                                |
| case      | unit test  | single function               | function                  |                                |
|           | SIT        | components                    | multi steps               |                                |
|           | E2E        | workflow                      | SWE-bench                 |                                |

## BFCL

- Berkeley Function Calling Leaderboard (Gorilla team, UC Berkeley)
- object: LLM layer, tool / function calling ability
- categories: simple, multiple, parallel, irrelevance detection, multi-turn
- metric: AST accuracy + executable accuracy (non-python langs)
- note: pre-agent layer — measures calling, not solving

## SWE-bench

### lineage / 2026 generation change

| benchmark              | owner              | status                 | scale                    | note                                                     |
| ---------------------- | ------------------ | ---------------------- | ------------------------ | -------------------------------------------------------- |
| SWE-bench (full)       | Princeton          | legacy                 | 2,294                    | real GitHub issues, python only                           |
| SWE-bench Verified     | OpenAI + Princeton | **retired ~2026-02**   | 500                      | audit: ~59% of hard tasks have test/description defects   |
| SWE-bench Pro          | Scale AI           | current                | 1,865 tasks / 41 repos   | contamination-resistant; Verified 80-96% -> Pro ~60%      |
| Terminal-Bench         | Laude / Stanford   | current, v4.0 2026-08  | 66 tasks                 | 4 generations in one year; cross-version score invalid    |
| FrontierCode 1.1       | Cognition          | current, 2026-10-12    | -                        | OSS maintainer judges "would you merge this diff"         |
| Next.js Agent Evals    | Vercel             | live, rerun 2026-10-13 | -                        | real Next.js repo tasks, agent-level not model-level      |
| ANCHOR                 | arXiv 2610.16859   | new, 2026-10-12        | 358 tasks                | engineering constraint following, not just correctness    |
| Doc2Harness            | arXiv 2610.20344   | new                    | -                        | long-horizon harness engineering, deterministic            |
| AgentBeats             | arXiv 2606.13608   | new                    | 298 judge / 467 subject  | AAA: benchmark as judge agent, A2A + MCP; N*M -> N+M      |

- reporting rule now: every score carries (version, source, date); cross-version comparison is invalid
- Verified saturation numbers are frozen/archived: Claude Opus 5 97.0%, DeepSeek V4 Pro 0813 96.4%

### scoreboard (2026-10, second-hand — verify against primary board)

| benchmark           | leader                        | score                                  |
| ------------------- | ----------------------------- | -------------------------------------- |
| SWE-bench Pro       | Claude Opus 5.5 (vendor)      | 89.9%; Fable 5.1 81.2%                 |
| Terminal-Bench 4.0  | GPT-6 Astra / Claude Fable 5.1| 58.18% / 57.88%; Opus 5.5 self-report 66.4% |
| FrontierCode 1.1    | Claude Opus 5.5 (max)         | 52% score / 58% pass — still >40% fail |
| Next.js Agent Evals | Opus 5.5 (high), GPT 6 Sol    | both 97%                               |
| ANCHOR              | Opus 4.8 + Cline              | 70.1% atomic / 63.7% relation          |

## general / long-horizon agent

| benchmark              | source             | setting                                       | top result                                  |
| ---------------------- | ------------------ | --------------------------------------------- | ------------------------------------------- |
| AgencyBench            | ACL 2026           | 6 capabilities / 32 scenarios / 138 tasks, ~90 tool calls, 1M tokens | closed 48.4% vs open 32.1%   |
| AgentGym2              | ACL 2026           | de-idealized env: noise, underspecified input | even Gemini / GPT-5 struggle                |
| EdgeBench              | -                  | 134 tasks x 12h+ interaction                  | Claude Opus 4.8 ~51% @12h; log-sigmoid scaling |
| Claw-Eval-Live         | arXiv 2604.28139   | live workflows, quarterly refresh             | best 66.7%                                  |
| OSWorld 2.0            | XLANG Lab          | 108 desktop workflows, ~1.6 human-hours each  | GPT-6 Astra 72.6 / Opus 5 70.6 / Gemini 4 Argon 69.2; best OSS Kimi K3 58.3 |
| RecreationBench        | arXiv 2609.22000   | rebuild running app on 5 OS platforms         | Astra 58.1% partial but only 2.8% pass all tests |
| EmbodiedSkillBench     | arXiv 2610.16238   | use of embodied skill packages (250 skills)   | -                                           |
| GAIA (HAL)             | Princeton          | scaffolded vs bare-model leaderboard          | Sonnet 4.5 74.6% scaffolded vs GPT-5 Mini 44.8% bare |
| Agents' Last Exam      | 2026-10-08 board   | -                                             | GPT-6 Astra 59.3%                           |

## industry leaderboards

- Artificial Analysis
  - AA-Briefcase (2026-06): frontier knowledge work; Fable 5 leads rubric pass, but all-criteria pass only 3% of tasks
  - EnterpriseOps-Gym-AA (2026-07-08): 8 domains, SQL-verified outcomes; Fable 5 51% / Gemini 3.5 Flash 50% / best open GLM-5.2 43%
  - also hosts Terminal-Bench, FrontierCode runs
- JobBench (2026-10-06): 35 occupations; Muse Spark 1.3 64.9% / MiMo-V2.6-Pro 62.0%
- AutomationBench (2026-10-08): 22 models, nearing saturation; DeepSeek V4.1 Flash 54.8%

## takeaways

1. not comparable across version / harness — GAIA shows 20-60 pts spread from scaffold choice alone; AgentBeats finds harness effect often > model gap
2. long-horizon pass rate still only 50-67% for frontier agents
3. strict binary << partial credit — RecreationBench 58.1% partial vs 2.8% all-tests-pass; OSWorld 2.0 outright completion ~20.6%
4. duration decay: 396 tasks x 3 domains x 4 duration buckets, mean pass@1 76.3% (short) -> 52.1% (very long), -24.3 pts
5. dimensions expanding beyond function: constraint following (ANCHOR), cost / efficiency, reliability (pass^k)
6. declare the statistic — partial credit and strict binary tell different stories

## sources

- https://artificialanalysis.ai/articles/frontiercode
- https://nextjs.org/evals
- https://arxiv.org/abs/2610.16859 (ANCHOR)
- https://arxiv.org/html/2610.20344v1 (Doc2Harness)
- https://www.alphaxiv.org/abs/2606.13608 (AgentBeats)
- https://llmlearner.com/rankings/osworld-2
- https://benchlm.ai/benchmarks/agentslastexam
- https://presenc.ai/research/terminal-bench-leaderboard-2026
