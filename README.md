# Modeling the Temperature-dependent Size Change of Polydisperse Nano-Objects using a Deep Generative Model
Our study introduces a pioneering algorithm that harnesses the density distribution of individual microgels to construct a clustering space, aimed at identifying the most representative samples. By leveraging the density distribution, we gain deeper insights into the microgel structure, facilitating more effective grouping. Within each cluster, we identify the top N samples that best represent the group, while filtering out any noisy samples. This method streamlines microgel analysis and enhances our capacity to extract valuable insights from intricate datasets. Additionally, we expand upon this concept by incorporating a deep generative model to simulate microgel behavior across varying temperatures.





## Sample Selection Overview
![Density based clustering](https://github.com/rezazad68/Microgel_generative/blob/main/images/sample_selection_process-1.png)



## Deep Generative Overview
![Density based clustering](https://github.com/rezazad68/Microgel_generative/blob/main/images/microgel_method_cvae.png)

#### Run
Please execute "sample_selection_demo.m" file to observe the algorithm in action for sample selection. </br>
Please run "generative_method" for training the CVAE model. the notebook file includes the detailed information.



