#python -m venv projectenv
=====================================================================
follow bellow steps to active  the virtual environment and next steps
=====================================================================

### Step 1: Activate the virtual environment

On Windows:
projectenv\Scripts\activate

On macOS and Linux:
source projectenv/bin/activate

### step 2: go to virtual environment

cd  projectenv

### Step 3:  Install the required packages

pip install -r requirements.txt

###  if you want to  Update the requirements.txt File

pip freeze > requirements.txt

====================
run the application 
====================

uvicorn main:app --reload 

=======================
go to end point  to check the application
=======================
use /docs

eg:
http://127.0.0.1:8000/docs
