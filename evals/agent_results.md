# Agent eval

9 scripted conversations, 14 turns, 39 checks, model `openai/gpt-oss-120b`, 866 listings (frozen snapshot). One run.

**38 of 39 checks passed (97%).**

| Conversation | Turn | Check | Result | Tools called |
|---|---|---|---|---|
| remote only | 1 | saves remote_only | pass | save_preference → search_jobs |
| remote only | 1 | searches the index | pass | save_preference → search_jobs |
| remote only | 1 | every job shown is remote | pass | save_preference → search_jobs |
| remote only | 1 | writes an answer | pass | save_preference → search_jobs |
| remote only | 1 | names a job it retrieved | **FAIL** | save_preference → search_jobs |
| remote only | 1 | answers in plain text | pass | save_preference → search_jobs |
| remote only | 2 | still only remote jobs, a turn later | pass | save_preference → search_jobs |
| remote only | 2 | writes an answer | pass | save_preference → search_jobs |
| avoid a keyword | 1 | saves 'sales' as something to avoid | pass | save_preference → search_jobs |
| avoid a keyword | 1 | no job shown mentions sales | pass | save_preference → search_jobs |
| avoid a keyword | 1 | writes an answer | pass | save_preference → search_jobs |
| two preferences at once | 1 | saves remote_only | pass | save_preference → save_preference → search_jobs |
| two preferences at once | 1 | saves 'crypto' as something to avoid | pass | save_preference → save_preference → search_jobs |
| two preferences at once | 1 | searches the index | pass | save_preference → save_preference → search_jobs |
| two preferences at once | 1 | names a job it retrieved | pass | save_preference → save_preference → search_jobs |
| location | 1 | saves the location | pass | save_preference → search_jobs |
| location | 1 | searches the index | pass | save_preference → search_jobs |
| location | 1 | writes an answer | pass | save_preference → search_jobs |
| a preference is changed | 1 | saves remote_only | pass | save_preference |
| a preference is changed | 2 | turns remote_only off again | pass | save_preference |
| no preference stated | 1 | saves no preference that was not stated | pass | search_jobs |
| no preference stated | 1 | searches the index | pass | search_jobs |
| no preference stated | 1 | writes an answer | pass | search_jobs |
| no preference stated | 1 | names a job it retrieved | pass | search_jobs |
| no preference stated | 1 | answers in plain text | pass | search_jobs |
| hide a job by name | 1 | searches the index | pass | search_jobs |
| hide a job by name | 1 | names a job it retrieved | pass | search_jobs |
| hide a job by name | 2 | hides exactly that job | pass | dismiss_job → search_jobs |
| hide a job by name | 2 | hides it without searching for it first | pass | dismiss_job → search_jobs |
| hide a job by name | 3 | the hidden job does not come back | pass | search_jobs |
| hide a job by name | 3 | hides nothing else | pass | search_jobs |
| hide a job by name | 3 | writes an answer | pass | search_jobs |
| details of one job | 1 | searches the index | pass | search_jobs |
| details of one job | 2 | reads that job with get_job | pass | get_job |
| details of one job | 2 | does not search again for it | pass | get_job |
| details of one job | 2 | writes an answer | pass | get_job |
| off topic | 1 | saves no preference that was not stated | pass | - |
| off topic | 1 | hides nothing | pass | - |
| off topic | 1 | replies | pass | - |
