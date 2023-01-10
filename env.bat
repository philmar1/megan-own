set CONDA_ENV_NAME=megan
set PROJECT_ROOT=%cd%
set DATA_DIR=%PROJECT_ROOT%\data
set CONFIGS_DIR=%CD%\configs
set LOGS_DIR=%CD%\logs
set MODELS_DIR=%CD%\models
set RANDOM_SEED=132435
echo "Project root set as %PROJECT_ROOT%"

CALL conda activate %CONDA_ENV_NAME%

set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
echo PROJECT_ROOT is "%PROJECT_ROOT%"
echo PYTHONPATH is "%PYTHONPATH%"

set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

:: set this variable to use other GPU than 0
set CUDA_VISIBLE_DEVICES=0

:: number of jobs in multithreaded parts of code. -1 <=> all available CPU cores
set N_JOBS=-1