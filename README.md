# Computational Physics and Machine Learning: Project 2

This repository contains scripts to solve partial differential equations with the Euler method and neural networks.
Moreover, a neural network algorithm is used to solve an eigenvalue problem. This repository was setup for project 2 of the lecture Computational Physics and Machine Learning.
## More information
* [Task Description](documentation/project_task.pdf)
* Scripts for Euler Solver can be found [here](PDE_Solver_Euler)
* Scripts for Neural Network Solver can be found [here](PDE_Solver_Neural)
* Scripts for Eigenvalue Problem Solver can be found [here](Eigen_Solver)

## How to setup (for Euler Solver)
1. Create conda environment
```python
conda create --name compsci_project_2 --file requirements.txt
```
2. Run one of the following lines depending on your use case.
```python
# if you only want to run the code
python setup.py install
# if you want to change some scripts later
python setup.py develop
```

## Contributors (alphabetic order)
* [Amandine Kaiser](https://www.mn.uio.no/compsci/english/people/phd-candidates/kaiser.html)
* [Anthony Val C. Camposano](https://www.mn.uio.no/compsci/english/people/phd-candidates/camposano.html)
* [Harish Pruthviraj Jain](https://www.mn.uio.no/compsci/english/people/phd-candidates/jain.html)

## Acknowledgement
All contributors are part of the [CompSci](https://www.mn.uio.no/compsci/english/) doctoral program which is managed by the Faculty of Mathematics and Natural Sciences at the University of Oslo (UiO).
The program is partly funded by the EU Horizon 2020 under the Marie Skłodowska-Curie Action (MSCA) - Co-funding of Regional, National and International Programmes (COFUND).