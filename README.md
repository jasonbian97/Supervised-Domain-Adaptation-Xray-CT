
## Project Structure

**planning**: Contains miscellaneous files. It could be useful papers, vague thoughts, etc.

**data**: 

- raw. Raw data files. e.g. soft-link to my local ~/AllDataSets
- cache: Preprocessed datasets that don’t need to be re-generated every time you perform an analysis.
- munge: Preprocessing data munging code, the outputs of which are put in cache.

**config**: Contains parameter files to run scripts, e.g. .json,.yaml files. I can adjust hyperparam here. 

**src**:

- scripts. All the scripts. e.g. train.py, test.py
- lib: Helper library functions. 

**results**: 

- **logs**: Output of scripts and any automatic logging. e.g. tensorboard output.

- **reports**: Output reports and content that might go into reports such as tables. e.g. it contains Latex project.

**README**: Notes that orient any newcomers to the project.

**TODO**: list of future improvements and bug fixes you plan to make.
