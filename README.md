# Welcome to Deepest Histology

## What is this?

This is an open source platform for end-to-end artificial intelligence (AI) in
computational pathology. It will enable you to use AI for prediction of any
"label" directly from digitized pathology slides. Common use cases which can be
reproduced by this pipeline are:

- prediction of microsatellite instability in colorectal cancer (Kather et al.,
  Nat Med 2019)
- prediction of mutations in lung cancer (Coudray et al., Nat Med 2018)
- prediction of subtypes of renal cell carcinoma (Lu et al., Nat Biomed Eng
  2021)
- other possible use cases are summarized by Echle et al., Br J Cancer 2021:
  https://www.nature.com/articles/s41416-020-01122-x

By default, the user of this pipeline can choose between different AI
algorithms, while the pre/post-processing is unchanged:

- vanilla deep learning workflow (Coudray et al., Nat Med 2018)
- vanilla Multiple instance learning (Campanella et al., Nat Med 2019)
- CLAM (Lu et al., Nat Biomed Eng 2021)

This pipeline is modular, which means that new methods for pre-/postprocessing
or new AI methods can be easily integrated. 

# Prerequisites

We use Deepest Histology on a local workstation server with Ubuntu 20.04 or
Windows Server 2019 and a CUDA-enabled NVIDIA GPU. The following packages are
required

[MARKO PLEASE ADD REQUIREMENTS]
