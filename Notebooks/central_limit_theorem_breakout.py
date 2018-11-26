## Central Limit Theorem test:

# 1. We first import the necessary libraries: 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 2. Next, create 100 simulations of 10 dice rolls, and in each simulation, find the average of the dice outcome (100 simulations of dice rolls in total):
n = 100

# In each simulation, there is one trial more than the previous simulation:
avg = []
# ... your code here

# 3. Function that will plot the histogram of the above generated values, 
#    where current is the latest figure:
def clt(current):
    # if animation is at the last frame, stop it
    plt.cla()
    if current == 100: 
        a.event_source.stop()

    plt.hist(avg[0:current])

    plt.gca().set_title('Expected value of dice rolls')
    plt.gca().set_xlabel('Average from dice roll')
    plt.gca().set_ylabel('Frequency')

    plt.annotate('dice roll = {}'.format(current), [3,27])
    
# 4. Using the animation function we can visualize how the histogram slowly resembles a normal distribution:
fig = plt.figure()
a = animation.FuncAnimation(fig, clt, interval=1)

plt.show()