# Research tracker

**Final goal:** SOTA fine-tuning/distillation method for a small model based on uncertainty estimation

**Research questions:**

- [ ] Is SFT beneficial for the subsequent CoT distillation?
- [ ] Can we find a data subset that contributes to the majority of the gains?
- [ ] Is all data beneficial to the final result or do we have some pieces of data that are harmful?
- [ ] What metrics could we use as a complexity score? How do they compare?
- [ ] How could we utilize knowledge the ground truth?
- [ ] What is the most efficient way to do large model distillation? How much does the starting point matter?
- [ ] Given that we find the part of the dataset that contributes the most to the final result, does it change with each epoch of training?
- [ ] Could we utilize complexity of the teacher model to curate the dataset for the student model?
- [ ] How can we find the theoretical explanation for empirical complexity windows?

- [Research tracker](#research-tracker)
  - [Progress](#progress)
    - [Is SFT beneficial for the subsequent CoT distillation?](#is-sft-beneficial-for-the-subsequent-cot-distillation)
    - [Can we find a data subset that contributes to the majority of the gains?](#can-we-find-a-data-subset-that-contributes-to-the-majority-of-the-gains)
    - [Is all data beneficial to the final result or do we have some pieces of data that are harmful?](#is-all-data-beneficial-to-the-final-result-or-do-we-have-some-pieces-of-data-that-are-harmful)
    - [What metrics could we use as a complexity score? How do they compare?](#what-metrics-could-we-use-as-a-complexity-score-how-do-they-compare)
    - [How could we utilize knowledge the ground truth?](#how-could-we-utilize-knowledge-the-ground-truth)
    - [What is the most efficient way to do large model distillation? How much does the starting point matter?](#what-is-the-most-efficient-way-to-do-large-model-distillation-how-much-does-the-starting-point-matter)
    - [Given that we find the part of the dataset that contributes the most to the final result, does it change with each epoch of training?](#given-that-we-find-the-part-of-the-dataset-that-contributes-the-most-to-the-final-result-does-it-change-with-each-epoch-of-training)
    - [Could we utilize complexity of the teacher model to curate the dataset for the student model?](#could-we-utilize-complexity-of-the-teacher-model-to-curate-the-dataset-for-the-student-model)
    - [How can we find the theoretical explanation for empirical complexity windows?](#how-can-we-find-the-theoretical-explanation-for-empirical-complexity-windows)
  - [Notes](#notes)
  - [Tasks](#tasks)
    - [Feb 13](#feb-13)


## Progress

### Is SFT beneficial for the subsequent CoT distillation?

**Answer:** TBD

**Ongoing experiments:**
-  Run SFT with CoT evals on 1, 2, 3, 4, 6, 8, 10, 15, 20 epochs
   -  [ ] MMLU-Pro
      -  [ ] Phi-4-mini
      -  [ ] Qwen 3B
      -  [ ] LLama 3B

**Completed experiments:**

### Can we find a data subset that contributes to the majority of the gains?

### Is all data beneficial to the final result or do we have some pieces of data that are harmful?

### What metrics could we use as a complexity score? How do they compare?

### How could we utilize knowledge the ground truth?

**Answer:** TBD

**Ongoing experiments:**
- Create a synthetic variations of the distilled CoT - ask model to explain the known answer, ask model to correct previous mistake knowing the true answer
  - [x] Create raw datasets
  - [ ] Clean datasets

**Completed experiments:**

### What is the most efficient way to do large model distillation? How much does the starting point matter?

### Given that we find the part of the dataset that contributes the most to the final result, does it change with each epoch of training?

### Could we utilize complexity of the teacher model to curate the dataset for the student model?

### How can we find the theoretical explanation for empirical complexity windows?

## Notes

1. Could we tie EIG to entropy windows? See BED-LLM


## Tasks

### Feb 13

Semyon:
- PR with cleaned synth
- PR with draft experiment: branch A, branch B 

Daniel:
- PR with CoT evals scipts - evals on 1,2,3,4,6,8,10,15,20
- PR with distillation with reasoning tokens script

Andrey:
- Refactor infra to easily swap datasets
- Find 2 new datasets and entropy splits