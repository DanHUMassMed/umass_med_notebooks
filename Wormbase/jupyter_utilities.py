import time

def formatted_elapsed_time(start,end=None):
    minute=60
    hour  =60 * minute

    if end == None:
        end = time.time()
    total_seconds = end - start
    hours = total_seconds // hour
    minutes = (total_seconds % hour) // minute
    seconds = (total_seconds % hour) % minute
    return f'Time: {hours=} {minutes=} {seconds=:.2f}'