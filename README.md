# Math Methods Lab 1: Excel Analysis Project

## Description

This project performs various analyses on Excel files using Python, including data cleaning, missing value handling, correlation analysis, and visualization. It processes data from `result.xlsx` and generates multiple outputs such as filled datasets, correlation matrices, heatmaps, and statistical summaries.

## Prerequisites

- Python 3.8 or higher (download from [python.org](https://www.python.org/downloads/))
- pip (usually included with Python; update with `python -m pip install --upgrade pip`)
- Git (for cloning the repository; download from [git-scm.com](https://git-scm.com/))

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/math_methods_lab_1.git
cd math_methods_lab_1
```

Replace `yourusername` with your GitHub username if the repository is hosted there.

### Step 2: Set Up Virtual Environment

Create a virtual environment to isolate dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

Deactivate with `deactivate` when done.

### Step 3: Install Dependencies

Install required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you encounter permission errors on Windows, run the command prompt as administrator or use `--user` flag: `pip install --user -r requirements.txt`.

## Usage

1. Ensure `result.xlsx` is in the project root directory.
2. Run the desired script(s):
   - `python analyze_excel.py` - Basic analysis
   - `python correlation_analysis.py` - Correlation analysis
   - `python fill_missing_values.py` - Fill missing values
   - Other scripts as needed for specific analyses
3. Outputs will be saved in the project directory (e.g., CSV files, Excel files, PNG images).

## Project Structure

- `analyze_excel.py` - Basic Excel analysis
- `correlation_analysis.py` - Correlation matrix and heatmap generation
- `fill_missing_values.py` - Missing value imputation
- `filter_low_filled_columns.py` - Column filtering based on fill rate
- `missing_values_stats.py` - Missing value statistics
- `text_columns_filter.py` - Text column handling
- `requirements.txt` - Python dependencies
- `.gitignore` - Files to ignore in Git
- Various output files (CSV, Excel, PNG)

## Outputs

- **Console**: Analysis results, statistics, and progress messages
- **Files**:
  - `summary.csv` - Statistical summaries
  - `correlation_matrix.csv` / `.xlsx` - Correlation data
  - `correlation_heatmap.png` - Visual heatmap
  - `result_filled.xlsx` - Dataset with filled missing values
  - Other processed datasets and visualizations

## Troubleshooting

- **Python not found**: Ensure Python is installed and added to PATH
- **Pip issues**: Update pip or use `python -m pip` instead of `pip`
- **Virtual environment issues**: Recreate venv if activation fails
- **Missing modules**: Reinstall requirements or check Python version compatibility
- **File not found errors**: Ensure input files are in the correct directory

## Contributing

1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Create a Pull Request

## License

This project is open-source. Please check the license file if included.
