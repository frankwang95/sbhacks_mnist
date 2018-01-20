# MNIST With Not-So-Deep Neural Networks
##### UCSB Hacks 2018, LogMeIn Machine Learning Workshop

## Setup Instructions
1. Clone the workshop resources from Github with `git clone https://github.com/frankwang95/sbhacks_mnist`
2. Get the Anaconda Python distribution.
    1. Download:
        - Mac: `curl https://repo.continuum.io/archive/Anaconda3-5.0.1-MacOSX-x86_64.sh -o anaconda_install.sh`
        - Linux: `wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda_install.sh`
        - Windows: Come back after you install Linux.
    2. Install:

```
chmod +x anaconda_install.sh
./anaconda_install.sh
```

3. Create a new virtual environment and switch to it.

```
conda create --name mnist --clone root
source activate mnist
```

4. Install Tensorflow and Plotly with `pip install tensorflow plotly`
5. Start a Jupyter Notebook with `jupyter notebook`
6. Hit up `localhost:8888` in your browser if it doesn't go there itself.
