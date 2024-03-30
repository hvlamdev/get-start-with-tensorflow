# Install the neccessary dependencies.

pip install google-cloud-logging

pip install ---upgrade protobuf

pip install --upgrade tensorflow

# Check python version
python --version

# Check if TensorFlow is installed. Run the following command in the terminal:
python -c "import tensorflow;print(tensorflow.__version__)"

# Using model
python model.py