"""
A pool that distributes tasks over a set of MPI processes. MPI is an API for
distributed memory parallelism.  This pool will let you run emcee without
shared memory, letting you use much larger machines with emcee.

The pool only support the "map" method at the moment.

See the example below, which should be run using mpirun with a number of
processes greater than two. They will not work in an interactive interpreter.
e.g.
mpirun -np 3 mpi_test.py

This pool is fairly general and not tailored to emcee.  I have a variant which
is more efficient for emcee but less general (and it is also a bit more
confusing - as it assumes that the same function is used for each map).

I have done no tests about the efficiency of this in terms of cpu usage, but it
should be good if the number of processes is much smaller than the number of
walkers, or a multiple of it and the time taken per log-prob is relatively
homogeneous.

Joe Zuntz.


A GENERAL  EXAMPLE:
####################
import mpi_pool

def function(x):
	return 2*x

pool = mpi_pool.MPIPool(debug=False)
result = pool.map(function, [2.3, 4, 5, "New York ", 99.8])
if pool.is_master(): print result
pool.close()
####################


AN EMCEE EXAMPLE:
#################
#To use this in emcee you need to be a little more careful, since you only want
#the master to run the sampler itself:
import mpi_pool
import emcee
import numpy as np

pool = mpi_pool.MPIPool(debug=False)
nwalkers = 50
ndim = 10
p0 = np.random.rand(nwalkers,ndim)

def log_prob(p):
	#A trivial Gaussian
	return -(p**2/2).sum()

if pool.is_master():
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
	for sample in sampler.sample(p0, iterations = 100):
		print sample[0]
else:
	pool.wait()
pool.close()



"""
from mpi4py import MPI

__all__ = ["MPIPool"]

class ClosePoolMessage(object):
	def __repr__(self):
		return "<Close pool message>"

class FunctionWrapper(object):
	def __init__(self, function):
		self.function = function

def error_function(task):
	raise RuntimeError("Pool was sent tasks before being told what function to apply.")

class MPIPool(object):
	""" An MPI pool object.  Only supports map.  See module docstrings for more info."""
	def __init__(self, comm=MPI.COMM_WORLD, debug=False):
		self.comm = comm
		self.rank = comm.Get_rank()
		self.size = comm.Get_size() - 1
		self.debug = debug
		self.function = error_function
		if self.size==0:
			raise ValueError("Tried to create an MPI pool, but there was only one MPI process available.  Need at least two.")

	def is_master(self):
		return self.rank==0

	def wait(self):
		if self.is_master():
			raise RuntimeError("Master node told to await jobs")
		status = MPI.Status()
		while True:
			#Event loop.  Await instructions
			if self.debug: print "Worker %d waiting for task" % self.rank

			#Blocking receive to wait for instructions
			task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
			if self.debug: print "Worker %d got task %r with tag %d" % (self.rank, task, status.tag)

			#Check if message is special sentinel signalling end.  Stop loop if so.
			if isinstance(task,ClosePoolMessage):
				if self.debug: print "Worker %d told to quit" % self.rank
				break

			#Check if message is special type containing new function to be applied
			if isinstance(task,FunctionWrapper):
				self.function = task.function
				if self.debug: print "Worker %d replaced its task function with %r" % (self.rank, self.function)
				continue

			#If not a special message, just run the known function on the input and return it asnychronously
			result = self.function(task)
			if self.debug: print "Worker %d sending answer %r with tag %d" % (self.rank, result, status.tag)
			self.comm.isend(result, dest=0, tag=status.tag) #Return result async

	def map(self, function, tasks):
		ntask = len(tasks)

		#If not the master just wait for instructions.
		if not self.is_master():
			self.wait()
			return

		F = FunctionWrapper(function)
		#Tell all the workers what function to use
		for i in xrange(self.size):
			self.comm.isend(F, dest=i+1)

		#Send all the tasks off.  Do not wait for them to be received, just continue.
		for i,task in enumerate(tasks):
			worker = i%self.size + 1
			if self.debug: print "Sent task %r to worker %d with tag %d" % (task, worker, i)
			self.comm.isend(task, dest=worker, tag=i)
		results = []

		#Now wait for the answers
		for i in xrange(ntask):
			worker = i%self.size+1
			if self.debug: print "Master waiting for answer from worker %d with tag %d" % (worker, i)
			result = self.comm.recv(source=worker, tag=i)
			results.append(result)
		return results

	def close(self):
		#Just send a message off to all the pool members which contains the
		#special "ClosePoolMessage" sentinel
		if self.is_master():
			for i in xrange(self.size):
				self.comm.isend(ClosePoolMessage(), dest=i+1)
