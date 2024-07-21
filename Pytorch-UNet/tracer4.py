import sys
import os
import inspect
import linecache
import json
import numpy as np
import pandas as pd
from PIL import Image

form_arr = []
call_stack = []

def get_shape(value):
    try:
        if isinstance(value, (list, tuple)):
            return len(value),
        elif isinstance(value, np.ndarray):
            return value.shape
        elif isinstance(value, pd.DataFrame):
            return value.shape
        elif isinstance(value, Image.Image):
            return value.size  # PIL Images have size (width, height)
        elif hasattr(value, 'shape'):
            return value.shape
        elif hasattr(value, '__len__'):
            return len(value),
        else:
            return None
    except Exception as e:
        return None

def trace_functions(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        function_name = code.co_name
        filename = code.co_filename
        
        # Check if the function is part of the target script
        if not filename.startswith(target_script_directory):
            # Skip tracing and execution of functions from built-in modules
            return
        
        # Get function arguments with their types and shapes
        arg_info = inspect.getargvalues(frame)
        args_with_details = {}
        for arg in arg_info.args:
            arg_value = arg_info.locals[arg]
            arg_type = str(type(arg_value))
            arg_shape = get_shape(arg_value)
            args_with_details[arg] = {"type": arg_type, "shape": arg_shape}
        
        # Trace and print function details
        lineno = frame.f_lineno

        # Add function call to the call stack
        if len(call_stack) == 0 or lineno not in call_stack[-1]["executed_function_lines"]:
            if len(call_stack) > 0:
                call_stack[-1]['executed_function_lines'].add(lineno)
                form_arr.append(call_stack[-1])
            call_stack.append({
                'function_name': function_name,
                'filename': filename,
                'lineno': lineno,
                'args': args_with_details,
                'lines': [],
                'executed_lines': set(),
                'executed_function_lines': set(),
                'extra_calls': 0
            })
        elif lineno in call_stack[-1]["executed_function_lines"]:
            call_stack[-1]["extra_calls"] += 1

    elif event == 'line':
        code = frame.f_code
        filename = code.co_filename
        lineno = frame.f_lineno
        
        # Print executed lines from the target script
        if filename.startswith(target_script_directory):
            line = linecache.getline(filename, lineno).strip()
            if line and not line.startswith("#"):
                if call_stack:
                    if len(call_stack[-1]["lines"]) > 0 and call_stack[-1]["lines"][-1] != line:
                        if lineno not in call_stack[-1]['executed_lines']:
                            call_stack[-1]['lines'].append(line)
                            call_stack[-1]['executed_lines'].add(lineno)
                    elif len(call_stack[-1]["lines"]) == 0:
                        if lineno not in call_stack[-1]['executed_lines']:
                            call_stack[-1]['lines'].append(line)
                            call_stack[-1]['executed_lines'].add(lineno)

    elif event == 'return':
        if call_stack:
            # Capture the return value with its type and shape
            return_value = {"return_value": {"type": str(type(arg)), "shape": get_shape(arg)}}
            call_stack[-1].update(return_value)
            
            if call_stack[-1]["extra_calls"] == 0:
                form_arr.append(call_stack[-1])
                function_call = call_stack.pop()
            else:
                call_stack[-1]["extra_calls"] -= 1
    
    return trace_functions

if __name__ == "__main__":
    # Ensure command-line arguments are passed
    if len(sys.argv) < 2:
        print("Usage: python trace_script.py <target_script> [args...]")
        sys.exit(1)

    # Path to the target script and its directory
    target_script_file = os.path.abspath(sys.argv[1])
    target_script_directory = os.path.dirname(target_script_file)

    # Set up tracing
    sys.settrace(trace_functions)     

    # Prepare the arguments for the target script
    target_script_args = sys.argv[1:]
    sys.argv = target_script_args
    
    # Run the target script using exec
    with open(target_script_file, "rb") as file:
        exec(compile(file.read(), target_script_file, 'exec'))
    
    sys.settrace(None)  # Disable tracing after execution

    # Process and print the collected function call information
    if len(call_stack) > 0:
        form_arr.append(call_stack[-1])

    # Output the collected data to a JSON file
    with open('trace_calls.json', 'w') as json_file:
        json.dump(form_arr, json_file, indent=4)
