

import tensorflow as tf	
import matplotlib.pyplot as plt
GOAL = 100

states = tf.range(GOAL+1)

headProb = 0.4


policy = tf.variable(tf.zeros(GOAL+1, tf.float32))

stateValue = tf.variable(tf.zeros(GOAL+1, tf.float32))
stateValue = stateValue[GOAL].assign(1.0)

while True:
	delta = 0.0
	for state in states[1:GOAL]:
		actions = tf.range(min(state,GOAL-state)+1)
		actionReturns = []
		for action in actions:
			actionReturns.append(headProb* stateValue[state+action] + (1-headProb)*stateValue[state-action])
		newValue = tf.maximum(actionReturns)
		delta += tf.abs(stateValue[state] - newValue)
		stateValue = stateValue[state].assign(newvalue)
	if delta < 1e-9:
		break

for state in states[1:GOAL]:
	actions = tf.range(min(state,GOAL - state)+1)
	actionReturns = []
	for action in actions:
		actionReturns.append(headProb*stateValue[state + action] + (1-headProb)*stateValue[state-action]) 
	policy = policy[state].assign(actions[tf.argmax(actionReturns)])

plt.figure(1)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.plot(stateValue)
plt.figure(2)
plt.scatter(states, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.show()

