 ```bash
 #!/bin/bash
 echo "Setting up environment and launching Prosperity Backtester App..."

 # Check if Python 3 is installed
 if ! command -v python3 &> /dev/null
 then
     echo "Error: python3 could not be found. Please install Python 3.9+."
     exit 1
 fi

 # Check if pip is available
 if ! python3 -m pip --version &> /dev/null
 then
      echo "Error: pip not found. Ensure Python 3 installation includes pip."
      exit 1
 fi


 # Optional: Create and activate a virtual environment
 if [ ! -d "venv" ]; then
     echo "Creating virtual environment (venv)..."
     python3 -m venv venv
     if [ $? -ne 0 ]; then
         echo "Error: Failed to create virtual environment."
         exit 1
     fi
 fi

 echo "Activating virtual environment..."
 source venv/bin/activate

 # Install requirements
 echo "Installing required packages from requirements.txt..."
 python3 -m pip install -r requirements.txt
 if [ $? -ne 0 ]; then
     echo "Error: Failed to install requirements. Check requirements.txt and network connection."
     exit 1
 fi

 # Launch the Streamlit app
 echo "Launching Streamlit app (app.py)..."
 streamlit run app.py

 echo "App launched. Press Ctrl+C in the terminal to stop the app."

 # Keep script running if needed, or just let streamlit take over
 # wait # Uncomment if you want the script to wait for streamlit to exit
 ```