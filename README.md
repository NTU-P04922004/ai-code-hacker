# AI Code Hacker

This repo contains source code to solve coding competition problems (only [Meta Hacker Cup](https://www.facebook.com/codingcompetitions/hacker-cup) problems are supported now).

## Setup

### Install Required Packages
```bash
pip install -r requirements.txt
```

### Prepare Data

1. Download [2024 Round 2 Data](https://drive.google.com/file/d/1buaEbjurH_7DnEohKHBA1UggoYSJtUj0/view?usp=drive_link)
2. Extract Data
   ```bash
   mkdir path_to_save_data
   tar xvf contestData_2024_r2.tar -C path_to_save_data
   ```

## Run Code

### AutoGen Version
```bash
python run_autogen.py \
    problem_id \
    problem_name \
    data_path
```

### LangChain Version
```bash
python run_langchain.py \
    problem_id \
    problem_name \
    data_path
```

### Arguments
- **problem_id**: Identifier for a problem (can be anything).
- **problem_name**: Name for the problem (usually the name of the directory).
- **data_path**: Path to the problem data (the unzipped folder).

## Benchmark
(WIP)
