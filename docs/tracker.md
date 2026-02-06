# Research tracker

**Final goal:** SOTA fine-tuning/distillation method for a small model based on uncertainty estimation

**Research questions:**

- Is SFT beneficial for the subsequent CoT distillation?
- Can we find a data subset that contributes to the majority of the gains?
- Is all data beneficial to the final result or do we have some pieces of data that are harmful?
- What metrics could we use as a complexity score? How do they compare?
- How could we utilize knowledge the ground truth?
- What is the most efficient way to do large model distillation? How much does the starting point matter?
- Given that we find the part of the dataset that contributes the most to the final result, does it change with each epoch of training?
- Could we utilize complexity of the teacher model to curate the dataset for the student model?
- How can we find the theoretical explanation for empirical complexity windows?

## Status

## Tasks

Semyon:
- PR with cleaned synth
- PR with draft experiment: branch A, branch B 

Daniel:
- PR with CoT evals scipts - evals on 1,2,3,4,6,8,10,15,20
- PR with distillation with reasoning tokens script

Andrey:
- Refactor infra to easily swap datasets
- Find 2 new datasets and entropy splits


## Updates

1. Could we tie EIG to entropy windows? See BED-LLM

