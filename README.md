# Optimization Algorithms for Data Science   

## Torch implementation of Projected Gradient Descent and Franke-Wolfe algorithm for Optimization

This is a repository containing code for the design project - OPTIMIZATION ALGORITHMS FOR DATA SCIENCE, made by RISHABH KHANDELWAL (2021B3A71207G) under the supervision of Prof. Sravan Danda.   

Contained in the 'optimizers' folder are 5 files.      

pgd_optimizer.py and frankwolfe_optimizer.py are the Optimizer Classes for the two algorithms (Projected Gradient Descent and Franke-Wolfe Algorithm). These two classes have been designed taking inspiration from the SGD class which already exists in the torch library. The code in our classes is written to mirror the structure and functionality of the SGD class as closely as possible.    

optimizer.py is a base class for all optimizers. It serves as a Light-weight stand-in for 'torch.optim.optimizer', and only the symbols required by pgd_optimizer.py and frank_wolfe.py are implemented.  

pgd_unit_tests.py and fw_unit_tests.py contain the tests to check working and functionality of both modules that we have designed.   

To run these, simply download the whole Optimizers directory and run the two unit test files within it. 
