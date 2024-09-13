import sys

def erroe_message_detial(error,error_detial:sys):
    _,_,exc_tb=error_detial.exc_info()
    
    file_name=exc_tb.tb_frame.f_code.co_filename
    
    error_message=f"Error Accoured in {file_name} at {exc_tb.tb_lineno} the error is {str(error)}"

    return error_message

class CustomException(Exception):
    def __init__(self, erroe_message,error_detial: sys):
        super().__init__(erroe_message)
        self.error_message=erroe_message_detial(erroe_message,error_detial)
        
    def __str__(self):
        return self.error_message