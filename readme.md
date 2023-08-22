**GCTOOLS pipeline (August 2023)**

Developed by Teymoor Saifollahi, Kapteyn Astronomical Institute

A data-analysis pipeline for the detection and characterization of extragalactic globular clusters (or generally star clusters).

(*This pipeline also makes use of several scripts/functions that have been developed by others)

**Contact**

Contact me in case of problems, questions, feedback, or for contributing to this pipeline:
- saifollahi@astro.rug.nl (this might not be available in the future, postdoc life)
- teymur.saif@gmail.com (personal email address)

**Installation**
- in your working directory, execute the following command in the command line:
`git clone https://github.com/teymursaif/GCTOOLS.git`

- you probably need to also install several Python libraries such as `Astropy` and `Photutils` as well as packages `Galfit` (the executable is available in the repository), `SExtractor` and `SWarp`. you can install such libraries using `pip install <library-name>` in the command line. Please check with their official website how to install them if the `pip` command failed.

**Setup the input data**
- All the configuration of the pipeline is done in `modules/initialize.py`

- Make sure data is prepared for the analysis by keeping them in the `data_dir` parameter as defined in `modules/initialize.py`. Frames for a given galaxy should be in the following format (as an example for HST data of galaxy DF44 in F606W):
`DF44_F606W.science.fits` -> science frame (and stacked)
`DF44_F606W.weight.fits` -> weight map associated with the science frame

- Make sure you have PSF models stored in the `psf_dir` parameter as defined in `modules/pipeline_functions`. PSF model for a given filter should be given in the following format (as an example for HST F606W): `psf_F606W.fits`

- Make sure that you have adjusted all the necessary inputs in `modules/initialize.py`. There are two kind of inputs: the first group that you need to adjust based on your environment and study, and the second group that by default are set to a reasonable value and control the work of the pipeline. You may leave them as they are.

- fill in the targets in  the `TARGETS` (in the `modules/initialize.py` file) for the objects you would like to analyse (more information can be found inside that file). This parameter contains basic info about objects such as their name, rough coordinates, distance in Mpc, filters, and some comments regarding the type of the galaxy (e.g. nucleated or not-nucleated)

**Run the pipeline_functions**
- in your working directory and in the command line execute:
`python gc_pipeline.py`

**Example**
- there is already data on MATLAS-2019, a UDG known for its GCs). After setup the basic, you should be able to run the pipeline. Some output jpg files are already available in the output directory which you can look to get an idea of what to expect. There is data in 4 filters: HST/ACS in F606W and F814, VEGAS (VST survey) in g and i. There is more data of this object (but not included in the example)

**Notes**
- the current version (August 2023) of the pipeline does the first two steps of the desired analysis (out of 4 steps which will be available in the next versions). These two steps are (GCpipeline.png):

1. Sersic modelling of the galaxy to estimate Sersic parameters of objects
2. Source detection and photometry to make a source catalogue with photometry in all filters

- The other steps (steps 3 and 4, to be developed) are:

3. producing artificial GCs (at the distance of the galaxy and the given PSF) and measuring the completeness of source extractions, as well as assessing the compactness index of GC
4. Using compactness criteria based on simulations and identifying GCs in the data, apply colour selection, measuring their properties

![Screenshot](https://raw.githubusercontent.com/teymursaif/GCTOOLS/main/GCpipeline.png)
