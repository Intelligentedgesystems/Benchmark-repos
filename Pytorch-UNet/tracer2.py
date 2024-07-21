import sys
import os
import inspect
import linecache

form_arr = []
call_stack = []

# Open the file in write mode
output_file = open('trace_calls.txt', 'w')


def trace_functions(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        function_name = code.co_name
        filename = code.co_filename
        
        # Check if the function is part of the target script
        if not filename.startswith(target_script_directory):
            # Skip tracing and execution of functions from built-in modules
            return
        
        # Trace and print function details
        lineno = frame.f_lineno

        # Add function call to the call stack
        if(len(call_stack)==0 or lineno not in call_stack[-1]["executed_function_lines"]):
            if len(call_stack) > 0:
                call_stack[-1]['executed_function_lines'].add(lineno)
                output_file.write(str(call_stack[-1]) + '\n\n')
            call_stack.append({
                'function_name': function_name,
                'filename': filename,
                'lineno': lineno,
                'lines': [],
                'executed_lines': set(),
                'executed_function_lines': set(),
                'extra_calls': 0
            })
        elif(lineno in call_stack[-1]["executed_function_lines"]):
            call_stack[-1]["extra_calls"] = call_stack[-1]["extra_calls"]+1

    elif event == 'line':
        code = frame.f_code
        filename = code.co_filename
        lineno = frame.f_lineno
        
        # Print executed lines from the target script
        if filename.startswith(target_script_directory):
            line = linecache.getline(filename, lineno).strip()
            if line and not line.startswith("#"):
                if call_stack:
                    if(len(call_stack[-1]["lines"])>0 and call_stack[-1]["lines"][-1]!=line):
                        if lineno not in call_stack[-1]['executed_lines']:
                            call_stack[-1]['lines'].append(line)
                            call_stack[-1]['executed_lines'].add(lineno)
                    elif (len(call_stack[-1]["lines"])==0):
                        if lineno not in call_stack[-1]['executed_lines']:
                            call_stack[-1]['lines'].append(line)
                            call_stack[-1]['executed_lines'].add(lineno)

    elif event == 'return':
        if call_stack:
            if(call_stack[-1]["extra_calls"]==0):
                output_file.write(str(call_stack[-1]) + '\n\n')
                function_call = call_stack.pop()
                form_arr.append(function_call)
            else:
                call_stack[-1]["extra_calls"] = call_stack[-1]["extra_calls"]-1
    
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

    # Close the output file
    output_file.close()

    # Process and print the collected function call information
    if(len(call_stack)>0):
        output_file.write(str(call_stack[-1]) + '\n\n')
