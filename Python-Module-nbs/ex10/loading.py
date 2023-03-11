import time
import sys

def ft_progress(lst):
    # Get the total number of items in the list
    total = len(lst)

    # Record the start time of the loop
    start_time = time.time()

    # Iterate through each item in the list
    for i, item in enumerate(lst):
        # Calculate the progress as a percentage
        progress = i / total

        # Calculate the elapsed time since the start of the loop
        elapsed_time = time.time() - start_time

        # Calculate the estimated time remaining until the loop is complete
        if progress > 0:
            eta = elapsed_time / progress - elapsed_time
        else:
            eta = 0

        # Convert the ETA to hours, minutes, and seconds
        minutes, seconds = divmod(eta, 60)
        hours, minutes = divmod(minutes, 60)

        # Convert the elapsed time to hours, minutes, and seconds
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
        elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)

        # Create a progress bar to display the progress of the loop
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length - 1)

        # Display the progress bar and other information
        sys.stdout.write('\rETA: {:02d}:{:02d}:{:02d} [{:3d}%][{}] {:5d}/{:5d} | elapsed time {:02d}:{:02d}:{:02d} ... '.format(
            int(hours), int(minutes), int(seconds),
            int(progress * 100), bar, i+1, total, int(elapsed_hours), int(elapsed_minutes), int(elapsed_seconds)))
        sys.stdout.flush()

        # Yield the current item to the caller
        yield item

    # Print a newline character after the loop is complete
    sys.stdout.write('\n')


listy = range(1000)
ret = 0
for elem in ft_progress(listy):
    ret += (elem + 3) % 5
    time.sleep(0.01)
print()
print(ret)