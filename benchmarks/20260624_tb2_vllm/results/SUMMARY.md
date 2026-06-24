# TB2-lite Replay Results (2026-06-24)

Primary ranking uses `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`.

| Rank | Model | Score | Cmd F1 | First Cmd Exact | Valid JSON | Template | Sec/Step | Load (s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | phase2-reasoning | 51.59 | 0.5193 | 50.8% | 76.2% | chat_template | 0.087 | 131.4 |
| 2 | toolbench | 51.46 | 0.5230 | 49.5% | 76.9% | chat_template | 0.087 | 134.9 |
| 3 | phase1-fabliq | 49.31 | 0.5022 | 47.2% | 76.6% | chat_template | 0.087 | 84.9 |
| 4 | mega-combined | 41.44 | 0.4081 | 42.9% | 55.4% | chat_template | 0.092 | 80.4 |
| 5 | phase1b-frombase | 40.88 | 0.3972 | 43.6% | 58.4% | chat_template | 0.092 | 80.5 |
| 6 | base | 40.51 | 0.3992 | 41.9% | 59.1% | chat_template | 0.093 | 81.1 |
| 7 | mega-lr5e7 | 40.06 | 0.4069 | 38.6% | 57.8% | chat_template | 0.093 | 80.9 |
| 8 | mega-lr2e6 | 39.68 | 0.3929 | 40.6% | 59.1% | chat_template | 0.092 | 80.7 |

Results directory: `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/benchmarks/20260624_tb2_vllm/results`
