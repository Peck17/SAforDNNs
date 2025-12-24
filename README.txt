Date: 
2025/12/24

Paper status: 
During the submission process.

Notification: 
This program is an ancillary component of the paper <Spectrum Analysis for Deep Neural Networks>, which has not yet been published. Open access to the authorization for this project is granted only after the paper has been officially published. The publication status of the paper will be updated in this document. 

Introduction:
1. "FSNanalysis.py", which is referred to by "CaliforniaHouse.py" and "FSNvalidation.ipynb", is a custom module for the paper.
2. "CaliforniaHouse.py" is used to generate the results of FSNs on the California housing price dataset. The results are recorded in the "FSNPerformance" folder.  "FSNPerformance" will be automatically created if it does not exist in the root directory of "CaliforniaHouse.py". 
3. "CaliforniaHouse.py" uses parallel computing, so please avoid using <Jupyter Notebook> to prevent running issues. During the interpolation operation, the program may encounter a deadlock, which might be caused by the GIL. If a deadlock happens, you can try rerunning it. Most of the time, the program runs normally.
4. "FSNvalidation.ipynb" is used to verify theories in signals and generates the graphs in the paper. After running the first code block, all the remaining code blocks in "FSNvalidation.ipynb" can run independently. The result images are saved in the "figure" folder. The recommended version of module <notebook> is 7.2.2.
5. "FSNPerformance" and "figure" contain the data that has already been generated. These data have already been used in the paper.

