import os

project_name = "laptop-price-predictor"

# List of files and directories to create
list_of_files = [

    # Data folders
    f"{project_name}/data/raw/laptop_data.xlsx",
    f"{project_name}/data/processed/preprocessed_data.csv",

    # Source code
    f"{project_name}/src/__init__.py",
    f"{project_name}/src/data/__init__.py",
    f"{project_name}/src/data/load_data.py",
    f"{project_name}/src/data/preprocess.py",

    f"{project_name}/src/models/__init__.py",
    f"{project_name}/src/models/train.py",
    f"{project_name}/src/models/evaluate.py",

    f"{project_name}/src/api/app.py",

    # Models
    f"{project_name}/models/best_model.pkl",

    # MLflow
    f"{project_name}/mlflow/mlruns/.gitkeep",

    # Tests
    f"{project_name}/tests/test_data.py",
    f"{project_name}/tests/test_models.py",

    # Notebooks
    f"{project_name}/notebooks/eda.ipynb",

    # Docker
    f"{project_name}/docker/Dockerfile",
    f"{project_name}/docker/docker-compose.yml",

    # GitHub Actions
    f"{project_name}/.github/workflows/ml_pipeline.yml",

    # Root files
    f"{project_name}/requirements.txt",
    f"{project_name}/config.yaml",
    f"{project_name}/README.md",
]

for filepath in list_of_files:
    filepath = os.path.normpath(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # Create file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            if filename.endswith(".py"):
                f.write("# Auto-generated file\n")
            elif filename.endswith(".md"):
                f.write(f"# {project_name}\n")
            else:
                pass

print("Project structure created successfully!")
