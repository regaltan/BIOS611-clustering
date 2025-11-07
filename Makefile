# Define the Python interpreter
PYTHON = python3

# Define the figure files to be created
TASK1_FIGURE = figures/gap_statistic_simulation.png
TASK2_FIGURES = figures/spectral_clustering_simulation.png figures/spectral_3d_plot.html

# Define the temporary Python scripts that will be created
TASK1_SCRIPT_PY = scripts/task_1_kmeans.py
TASK2_SCRIPT_PY = scripts/task_2_spectral.py

# --- Main Rule ---
# 'all' rule: The default rule, depends on all final figures
all: $(TASK1_FIGURE) $(TASK2_FIGURES)

# --- Task 1 Rule ---
# Creates the gap statistic plot
$(TASK1_FIGURE): scripts/task_1_kmeans.ipynb scripts/clusGap.py
	# Convert notebook to a script
	jupyter nbconvert --to script scripts/task_1_kmeans.ipynb
	# Run the temporary Python script
	$(PYTHON) $(TASK1_SCRIPT_PY)
	# Clean up the temporary script
	rm $(TASK1_SCRIPT_PY)

# --- Task 2 Rule ---
# Creates BOTH spectral clustering plots (the .png and the .html)
$(TASK2_FIGURES): scripts/task_2_spectral.ipynb scripts/clusGap.py
	# Convert notebook to a script
	jupyter nbconvert --to script scripts/task_2_spectral.ipynb
	# Run the temporary Python script
	$(PYTHON) $(TASK2_SCRIPT_PY)
	# Clean up the temporary script
	rm $(TASK2_SCRIPT_PY)

# --- Clean Rule ---
clean:
	# Remove generated figures
	rm -f figures/*.png figures/*.html
	# Remove temporary Python scripts
	rm -f scripts/*.py
	# Remove Jupyter/Anaconda cache folders
	rm -rf scripts/.ipynb_checkpoints scripts/.jupyter scripts/.virtual_documents scripts/anaconda_projects
	rm -rf .Trash-0 anaconda_projects