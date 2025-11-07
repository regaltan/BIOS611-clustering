# BIOS 611: Clustering Assignment

[cite_start]This project implements and analyzes K-means and Spectral Clustering in Python, following the requirements of the assignment[cite: 531].

## Project Structure

* `Dockerfile`: Defines the Python 3.10 environment with all necessary libraries.
* `requirements.txt`: Lists all Python packages (numpy, scikit-learn, plotly, etc.).
* [cite_start]`Makefile`: Automates the entire analysis[cite: 537].
* `scripts/`: Contains all Python code:
    * `clusGap.py`: The professor-provided Gap Statistic implementation.
    * `task_1_kmeans.ipynb`: Jupyter notebook for Task 1 (K-means).
    * `task_2_spectral.ipynb`: Jupyter notebook for Task 2 (Spectral Clustering).
* [cite_start]`figures/`: Contains all generated plots and figures[cite: 538].

## How to Run This Project

1.  **Build the Docker Image:**
    From the main project directory, run:
    ```bash
    docker build -t 611_python .
    ```

2.  **Run the Analysis:**
    The `Makefile` must be run from *inside* the container, as it requires `jupyter` to convert the notebooks.

    First, start an interactive shell in the container:
    ```bash
    docker run -it --rm -v "$(pwd)":/app 611_python bash
    ```
    
    Then, *inside the container's shell* (at the `root@...` prompt), run:
    ```bash
    make all
    ```
    This will execute both Jupyter notebooks and save all plots to the `figures/` folder.

3.  **Clean Up:**
    To remove all generated files, run `make clean` from inside the container.