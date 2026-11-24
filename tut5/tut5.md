# Tutorial 5
Remember, to install packages in the environment:
```
conda activate [environment path]
conda install jax
conda install numpyro
conda install corner
```
## What plots can you generate from the latest mentor code?
Should get fits for $M_h$ and $c$ and mass-to-light ratio $M/L$ from the NFW profile as in [Posti's blogpost](https://lposti.github.io/mlpq/posts/2022-11-02-gp_rotcurves.html) and accompanying [research note](https://iopscience.iop.org/article/10.3847/2515-5172/aca0df). We look at the non-Gaussian process.

For `rotmod.dat` I copied the entry from NGC2403 into a `.dat` file, by taking the rows for NGC2403 from `somegalaxies.txt` from Tutorial 2 & 3, or else you can go to the Newtonian models SPARC set [here](https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt).



