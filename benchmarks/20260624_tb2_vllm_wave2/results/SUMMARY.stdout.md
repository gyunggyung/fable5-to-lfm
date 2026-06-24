# TB2-lite Replay Results (2026-06-24)

Primary ranking uses `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`.

| Rank | Model | Score | Cmd F1 | First Cmd Exact | Valid JSON | Template | Sec/Step | Load (s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | mega-5ep | 41.47 | 0.4171 | 40.9% | 60.1% | chat_template | 0.091 | 43.1 |
| 2 | phase2b-frombase-reasoning | 41.22 | 0.4080 | 42.2% | 57.8% | chat_template | 0.093 | 43.8 |
| 3 | mega-10ep | 41.06 | 0.4083 | 41.6% | 57.8% | chat_template | 0.092 | 44.2 |
| 4 | mega-1ep | 40.43 | 0.4036 | 40.6% | 60.1% | chat_template | 0.092 | 44.6 |
| 5 | mega-phase2m-reasoning | 40.13 | 0.3993 | 40.6% | 58.7% | chat_template | 0.093 | 43.7 |
| 6 | fabliq-1.2b-mega | 16.43 | 0.1782 | 13.2% | 52.8% | chat_template | 0.062 | 31.5 |
| 7 | fabliq-1.2b | 15.95 | 0.1726 | 12.9% | 54.5% | chat_template | 0.061 | 31.6 |
| 8 | fabliq-1.2b-thinking | 5.62 | 0.0605 | 4.6% | 17.5% | chat_template | 0.073 | 31.7 |

Results directory: `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/benchmarks/20260624_tb2_vllm_wave2/results`
