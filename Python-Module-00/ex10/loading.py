import time

def ft_progress(lst):
    total = len(lst)
    start_time = time.time()
    
    for i, elem in enumerate(lst, start=1):
        elapsed_time = time.time() - start_time
        progress_percentage = int((i / total) * 100)
        progress_bar = "=" * (progress_percentage // 2)
        
        eta = elapsed_time / (i / total + 0.0001)
        
        yield elem
        
        print(f"ETA: {eta:.2f}s [{progress_percentage}%][{progress_bar:<50}] {i}/{total} | elapsed time {elapsed_time:.2f}s", end="\r")
    
    print()

listy1 = range(1000)
ret1 = 0
for elem1 in ft_progress(listy1):
    ret1 += (elem1 + 3) % 5
    time.sleep(0.01)

print(ret1)

listy2 = range(3333)
ret2 = 0
for elem2 in ft_progress(listy2):
    ret2 += elem2
    time.sleep(0.005)

print(ret2)
