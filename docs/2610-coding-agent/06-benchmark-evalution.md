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

## SWE-bench
