import sys
import time

total = 48000
for i in range(total):
    # … your heavy lifting here …

    # Print status, overwriting the same line each time
    msg = f"Completed iteration {i+1}/{total}"
    sys.stdout.write("\r" + msg)   # write carriage-return + message
    sys.stdout.flush()             # force it onto the screen

# when you’re done, print a newline so the prompt ends up on its own line
