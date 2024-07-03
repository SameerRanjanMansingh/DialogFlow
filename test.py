from src.data.my_logging_module import setup_custom_logger
import logging

# from my_logging_module import 

# Create a logger for your application
my_logger = setup_custom_logger("my_app", log_level=logging.INFO, log_file="model.log")

# Log messages
# my_logger.info("Application started")

# logger = my_logging_module.main(filename='model.log', level='info', when='D', backCount=3)


try:
    s*p
    my_logger.info("print successfully")
except Exception as e:
    print(f"An error occurred while building the model: {e}")
    my_logger.warning(f"An error occurred while building the model: {e}")
