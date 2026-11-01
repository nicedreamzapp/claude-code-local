| Model | Setup | Pass rate | Valid tool calls | Leaked (unparsed) calls | Invalid tool_use | Median wall s (all) | Median wall s (passed) | Median out tok/s | Timeouts | Peak server RSS / footprint |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-e4b | mlx | 20/30 (67%) | 142/144 (99%) | 2 | 0 | 19.9 | 17.0 | 33.8 | 1 | 3.5 / 8.8 GB |
| gemma4-e4b | mlx-fix | 27/30 (90%) | 185/185 (100%) | 0 | 0 | 21.4 | 20.3 | 34.9 | 3 | 7.6 / 8.7 GB |
| gemma4-e4b | ollama | 27/30 (90%) | 77/77 (100%) | 0 | 0 | 20.8 | 20.6 | 36.6 | 1 | 12.1 / 5.8 GB |
| gemma4-e4b | ollama-t02 | 29/30 (97%) | 81/81 (100%) | 0 | 0 | 20.4 | 20.0 | 35.1 | 0 | 12.1 / 4.7 GB |
| qwen35-2b | mlx | 28/30 (93%) | 165/166 (99%) | 1 | 0 | 9.1 | 8.8 | 60.3 | 1 | 2.3 / 3.4 GB |
| qwen35-2b | mlx-fix | 28/30 (93%) | 188/188 (100%) | 0 | 0 | 8.8 | 7.7 | 59.7 | 1 | 2.3 / 3.3 GB |
| qwen35-2b | mlx-raw | 29/30 (97%) | 167/175 (95%) | 2 | 6 | 9.8 | 9.3 | 55.6 | 1 | 2.3 / 3.4 GB |
| qwen35-2b | ollama | 25/30 (83%) | 320/322 (99%) | 0 | 2 | 23.2 | 23.3 | 52.8 | 0 | 7.4 / 9.5 GB |
| qwen35-2b | ollama-16k | 25/30 (83%) | 239/242 (99%) | 0 | 3 | 14.7 | 13.2 | 72.0 | 0 | 9.7 / 7.9 GB |
| qwen35-2b | ollama-t02 | 20/30 (67%) | 814/815 (100%) | 0 | 1 | 12.4 | 11.9 | 63.8 | 4 | 10.2 / 8.4 GB |
| qwen35-4b | mlx | 30/30 (100%) | 100/100 (100%) | 0 | 0 | 20.0 | 20.0 | 17.8 | 1 | 8.2 / 9.1 GB |
| qwen35-4b | mlx-raw | 30/30 (100%) | 130/130 (100%) | 0 | 0 | 28.3 | 28.3 | 16.4 | 0 | 8.3 / 9.0 GB |
| qwen35-4b | ollama | 30/30 (100%) | 118/118 (100%) | 0 | 0 | 28.6 | 28.6 | 21.0 | 2 | 12.3 / 4.7 GB |
| qwen35-4b | ollama-16k | 28/30 (93%) | 138/138 (100%) | 0 | 0 | 29.6 | 27.9 | 21.1 | 2 | 12.4 / 5.8 GB |
| qwen35-4b | ollama-t02 | 29/30 (97%) | 113/113 (100%) | 0 | 0 | 23.8 | 23.1 | 22.5 | 1 | 12.6 / 4.8 GB |

file_path arguments that reached Claude Code with the leading '/' dropped (written relative to the work dir, so the file lands in the wrong place):

| Model | Setup | Calls | Runs affected |
|---|---|---|---|
| gemma4-e4b | mlx | 0 | 0/30 |
| gemma4-e4b | mlx-fix | 0 | 0/30 |
| gemma4-e4b | ollama | 0 | 0/30 |
| gemma4-e4b | ollama-t02 | 0 | 0/30 |
| qwen35-2b | mlx | 0 | 0/30 |
| qwen35-2b | mlx-fix | 0 | 0/30 |
| qwen35-2b | mlx-raw | 6 | 5/30 |
| qwen35-2b | ollama | 30 | 15/30 |
| qwen35-2b | ollama-16k | 24 | 13/30 |
| qwen35-2b | ollama-t02 | 159 | 18/30 |
| qwen35-4b | mlx | 0 | 0/30 |
| qwen35-4b | mlx-raw | 0 | 0/30 |
| qwen35-4b | ollama | 0 | 0/30 |
| qwen35-4b | ollama-16k | 0 | 0/30 |
| qwen35-4b | ollama-t02 | 0 | 0/30 |

Malformed tool calls that reached Claude Code (from the transcripts). leak_* = tool-call markup left in plain text, never run; bad_name_* = a tool_use whose name is not a declared tool (case variant, or anything else such as an alias or Write). 'Failed runs with any' = runs that failed and contain at least one of these:

| Model | Setup | leak_gemma_full | leak_gemma_no_opener | leak_function_tag | leak_tool_call_json | bad_name_case | bad_name_other | Runs with any | Failed runs with any | Unknown names |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-e4b | mlx | 1 | 0 | 0 | 0 | 0 | 0 | 1/30 | 0 | - |
| gemma4-e4b | mlx-fix | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| gemma4-e4b | ollama | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| gemma4-e4b | ollama-t02 | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| qwen35-2b | mlx | 0 | 0 | 0 | 1 | 0 | 0 | 1/30 | 1 | - |
| qwen35-2b | mlx-fix | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| qwen35-2b | mlx-raw | 0 | 0 | 0 | 2 | 0 | 6 | 7/30 | 1 | Write x6 |
| qwen35-2b | ollama | 0 | 0 | 0 | 0 | 0 | 2 | 2/30 | 1 | Write x2 |
| qwen35-2b | ollama-16k | 0 | 0 | 0 | 0 | 0 | 2 | 2/30 | 0 | Write x2 |
| qwen35-2b | ollama-t02 | 0 | 0 | 0 | 0 | 0 | 1 | 1/30 | 1 | Write x1 |
| qwen35-4b | mlx | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| qwen35-4b | mlx-raw | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| qwen35-4b | ollama | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| qwen35-4b | ollama-16k | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |
| qwen35-4b | ollama-t02 | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 | 0 | - |

Per task (passes out of reps):

| Task | gemma4-e4b mlx | gemma4-e4b mlx-fix | gemma4-e4b ollama | gemma4-e4b ollama-t02 | qwen35-2b mlx | qwen35-2b mlx-fix | qwen35-2b mlx-raw | qwen35-2b ollama | qwen35-2b ollama-16k | qwen35-2b ollama-t02 | qwen35-4b mlx | qwen35-4b mlx-raw | qwen35-4b ollama | qwen35-4b ollama-16k | qwen35-4b ollama-t02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t01_write_file | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| t02_edit_value | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| t03_fix_failing_test | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| t04_run_and_record | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| t05_create_and_run | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 1/3 | 3/3 | 2/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| t06_rename_across_files | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| t07_edit_json | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| t08_search_and_count | 3/3 | 1/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| t09_fix_syntax_error | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| t10_package_and_test | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

claude-code-local server log, repair events (whole run set, warm-up included):

| Run set | Gemma call parsed | Gemma missing opener recovered | Gemma no name match | Gemma no args parsed | unrecoverable JSON | function-in-tag parsed | garbled JSON recovered | tagless call recovered | tool inferred from params | tool name case fixed | tool name alias | argument alias | path leading slash restored | Bash timeout s->ms | re-prompt retries | retries that succeeded | tool calls emitted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-e4b mlx | 39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 136 |
| gemma4-e4b mlx-fix | 188 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 188 |
| qwen35-2b mlx | 0 | 0 | 0 | 0 | 0 | 100 | 1 | 0 | 1 | 0 | 0 | 0 | 25 | 2 | 1 | 1 | 160 |
| qwen35-2b mlx-fix | 0 | 0 | 0 | 0 | 0 | 109 | 3 | 0 | 0 | 0 | 0 | 0 | 25 | 2 | 0 | 0 | 180 |
| qwen35-2b mlx-raw | 0 | 0 | 0 | 0 | 0 | 109 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 2 | 2 | 172 |
| qwen35-4b mlx | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 90 |
| qwen35-4b mlx-raw | 0 | 0 | 0 | 0 | 3 | 127 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 122 |

Against claude-code-local (mlx) on the same model: pass-rate difference with a two-sided Fisher exact p, and the per-task wall-time ratio (median over the 10 tasks of setup median / mlx median; above 1 = slower than mlx):

| Model | Setup | Passes vs mlx | Fisher p | Wall-time ratio vs mlx |
|---|---|---|---|---|
| gemma4-e4b | mlx-fix | 27/30 vs 20/30 | 0.06 | 0.98x |
| gemma4-e4b | ollama | 27/30 vs 20/30 | 0.06 | 1.05x |
| gemma4-e4b | ollama-t02 | 29/30 vs 20/30 | 0.01 | 1.02x |
| qwen35-2b | mlx-fix | 28/30 vs 28/30 | 1.00 | 0.99x |
| qwen35-2b | mlx-raw | 29/30 vs 28/30 | 1.00 | 1.24x |
| qwen35-2b | ollama | 25/30 vs 28/30 | 0.42 | 2.81x |
| qwen35-2b | ollama-16k | 25/30 vs 28/30 | 0.42 | 1.77x |
| qwen35-2b | ollama-t02 | 20/30 vs 28/30 | 0.02 | 1.82x |
| qwen35-4b | mlx-raw | 30/30 vs 30/30 | 1.00 | 1.19x |
| qwen35-4b | ollama | 30/30 vs 30/30 | 1.00 | 1.17x |
| qwen35-4b | ollama-16k | 28/30 vs 30/30 | 0.49 | 1.29x |
| qwen35-4b | ollama-t02 | 29/30 vs 30/30 | 1.00 | 0.97x |

Cloud guard CONNECT attempts: 0; non-local sockets seen on Claude Code: none
