import logging
import os

# Function to setup logging configuration
def setup_logging(log_file='etl_pipeline.log'):
    """Setup the logging configuration."""
    # Create the log directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        filename=log_file_path,  # Log file path
        level=logging.INFO,  # Log level (INFO, DEBUG, ERROR, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        filemode='w'  # Overwrite log file each time the pipeline is run
    )

    # Log that the logging system is set up
    logging.info("Logging setup complete.")

# Function to log error messages
def log_error(message, exc=None):
    """Log error message."""
    logging.error(message)
    if exc:
        logging.error(f"Exception: {exc}")

# Function to log general information
def log_info(message):
    """Log information message."""
    logging.info(message)

# Function to log warning messages
def log_warning(message):
    """Log warning message."""
    logging.warning(message)
