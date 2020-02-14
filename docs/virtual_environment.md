To set up a virtual environment:
- Install it: `pip install virtualenv`
- Create the virtual environment: `virtualenv --always-copy --system-site-packages --python=python3 .venv`
- Install the needed packages: `.venv/bin/pip install -q -r requirements.txt`

To use the virtual environment, enter it: `source .venv/bin/activate`

To exit the virtual environment use: `deactivate`

To delete the virtual environment just delete the `.venv` folder: `rm -r .venv`
