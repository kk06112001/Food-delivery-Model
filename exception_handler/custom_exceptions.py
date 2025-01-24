class ETLException(Exception):
    """Base class for all exceptions raised by the ETL pipeline."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class DataExtractionException(ETLException):
    """Exception raised for errors in the extraction phase."""
    def __init__(self, message="Error occurred during data extraction"):
        self.message = message
        super().__init__(self.message)

class DataTransformationException(ETLException):
    """Exception raised for errors in the transformation phase."""
    def __init__(self, message="Error occurred during data transformation"):
        self.message = message
        super().__init__(self.message)

class ModelTrainingException(ETLException):
    """Exception raised for errors in the model training phase."""
    def __init__(self, message="Error occurred during model training"):
        self.message = message
        super().__init__(self.message)

class InvalidDataFormatException(ETLException):
    """Exception raised when the data format is invalid."""
    def __init__(self, message="Invalid data format encountered"):
        self.message = message
        super().__init__(self.message)

class ModelEvaluationException(ETLException):
    """Exception raised for errors during model evaluation."""
    def __init__(self, message="Error occurred during model evaluation"):
        self.message = message
        super().__init__(self.message)
