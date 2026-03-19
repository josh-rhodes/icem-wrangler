# icem-wrangler
Reading and processing I-CeM data (https://www.campop.geog.cam.ac.uk/research/projects/icem/).

Some possibly helpful files and code for processing I-CeM Census data. I'm in the process of converting from pandas to the polars library (which is much faster and better at handling large datasets), so the code here is very much work in progress (though it should actually work!) and any comments/suggestions for improvements/optimisation are welcome!

Use `read_icem_efficient.ipynb` to read in the very large csv .txt census files and convert them to much smaller .parquet files while also fixing various issues with the raw census data.