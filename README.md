## Iris Classification Project - CI/CD Pipeline

This repository contains the "Iris homework" project, which demonstrates a complete MLOps pipeline for a simple machine learning model. The primary focus is on Continuous Integration (CI), data and model versioning with DVC, and automated reporting.

The project is configured to work with a `main` and `dev` branch structure, where changes are tested on `dev` before being merged into `main`.

### Core Features

*   **CI/CD with GitHub Actions**: The pipeline is defined in `.github/workflows/ci-main.yml` and automatically triggers on pushes to `dev` and `main`, and on pull requests to `main`.
*   **Data & Model Versioning with DVC**: The project uses DVC to manage the dataset and the trained model. The remote storage is configured on Google Cloud Storage (GCS). The CI pipeline automatically pulls the required artifacts using `dvc pull`.
*   **Automated Testing**: Unit tests for data validation and model evaluation are implemented using `pytest`. These tests are a mandatory step in the CI pipeline.
*   **CML-Powered Reporting**: After tests and a sanity inference run are complete, a report is automatically generated and posted as a comment on the corresponding commit or pull request using CML (Continuous Machine Learning).

### Pipeline Workflow

The CI pipeline (`.github/workflows/ci-main.yml`) executes the following steps:

1.  **Checkout Code**: Checks out the repository code.
2.  **Set up Python**: Configures the correct Python environment.
3.  **Install Dependencies**: Installs all required packages from `requirements.txt`.
4.  **Authenticate & Fetch Data**: Authenticates with Google Cloud and uses DVC to pull the versioned model and data from GCS.
5.  **Run Unit Tests**: Executes the test suite using `pytest` to ensure code and model quality. Test results are captured.
6.  **Run Sanity/Inference Test**: Runs `inference.py` to perform a sample prediction, ensuring the model loads and runs correctly. The output is captured.
7.  **Generate and Publish Report**: Uses CML to create a markdown report summarizing the test and inference results, then posts it as a comment on the GitHub commit or pull request.

### How to Use

#### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv .env
    source .env/bin/activate
    pip install -r requirements.txt
    ```
3.  **Run tests locally:**
    ```bash
    pytest
    ```

#### Interacting with the Pipeline

*   **Push to `dev` or `main` branch**: Pushing new commits to the `dev` or `main` branch will trigger the CI pipeline. You can monitor the run in the "Actions" tab of the GitHub repository. A CML report will be posted on the commit.
*   **Create a Pull Request**: Opening a pull request from `dev` to `main` will also trigger the CI pipeline. The CML report will appear in the pull request's comment thread, providing clear feedback before merging.