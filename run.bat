 ```batch
 @echo off
 echo Setting up environment and launching Prosperity Backtester App...

 REM Check if Python is installed and accessible
 python --version > nul 2>&1
 if %errorlevel% neq 0 (
     echo Error: Python not found in PATH. Please install Python 3.9+ and add it to PATH.
     pause
     exit /b 1
 )

 REM Check if pip is available
 pip --version > nul 2>&1
 if %errorlevel% neq 0 (
      echo Error: pip not found. Ensure Python installation includes pip.
      pause
      exit /b 1
 )

 REM Optional: Create and activate a virtual environment
 if not exist venv (
     echo Creating virtual environment (venv)...
     python -m venv venv
     if %errorlevel% neq 0 (
         echo Error: Failed to create virtual environment.
         pause
         exit /b 1
     )
 )

 echo Activating virtual environment...
 call venv\Scripts\activate

 REM Install requirements
 echo Installing required packages from requirements.txt...
 pip install -r requirements.txt
 if %errorlevel% neq 0 (
     echo Error: Failed to install requirements. Check requirements.txt and network connection.
     pause
     exit /b 1
 )

 REM Launch the Streamlit app
 echo Launching Streamlit app (app.py)...
 streamlit run app.py

 echo App launched. Closing this window will stop the app.
 pause
 ```