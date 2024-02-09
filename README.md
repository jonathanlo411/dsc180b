# DSC180 - Auditlab
<a href="https://github.com/jonathanlo411/dsc180b/releases"><img  src="https://img.shields.io/github/v/release/jonathanlo411/dsc180b"></a><a  href="https://github.com/jonathanlo411/dsc180b/blob/main/LICENSE"><img  src="https://img.shields.io/github/license/jonathanlo411/dsc180b"></a>
[![Alt](https://repobeats.axiom.co/api/embed/1b4b2b98c4e4c93342bc7f974c3e4d91b43087b7.svg "Repobeats analytics image for DSC180B")](https://github.com/jonathanlo411/dsc180b/pulse/monthly) 

## Overview
This is part 2 of the DSC180 sequence. For the first part, please see the [DSC180A repository](https://github.com/jonathanlo411/dsc180a). This quarter focuses on auditing toxicity models [TextBlob](https://textblob.readthedocs.io/en/dev/), [vaderSentiment](https://github.com/cjhutto/vaderSentiment), and [Perspective API](https://perspectiveapi.com/).

## Using
1. **Obtain an Perspective API Key**: For all cases you will need a Perspective API key. You can find instructions on how to do so from the [Perspective API Docs](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US).
2. **Add config information**: Fill out the Perspective API key into the `sample.secrets.json` and rename the file to `secrets.json`. For example usage, see `notebooks/model-exploration.ipynb`. 
3. **Setup your environment**: Install packages `TextBlob`, `vaderSentiment`, and `Google API Client`. These are in addition to the the standard packages provided by Jupyter.
```bash
pip install textblob vaderSentiment google-api-python-client
python -m textblob.download_corpora
```
4. **Run tests**: Run one of the notebooks under `notebooks/`. It is recommended to run one of the `model-audit-<NAME>.ipynb` notebooks as they are the most up-to-date.
