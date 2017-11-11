#!/usr/bin/env python3

class RDD:
	def __init__(self, iterator):
		self.data = iterator

	def collect(self):
		return self.data

	def first(self):
		return self.data[0]

	def top(self, num):
		return self.data[0:num]

	def persist(self):
		return self

	def cache(self):
		return self

	def map(self, func):
		return parallelize([func(x) for x in self.data])

	def reduce(self, func):
		s = self.data[0]
		for x in self.data[1:]:
			s = func(s,x)
		return s

	def mapPartitions(self, func, n = 3):
		l = len(self.data)
		return RDD([func(self.data[i*l//n : (i+1)*l//n]) for i in range(n)])


class PairRDD(RDD):
	def __init__(self, iterator):
		super(PairRDD, self).__init__(iterator)

	def keys(self):
		return RDD([x[0] for x in self.data])

	def values(self):
		return RDD([x[1] for x in self.data])

	def mapValues(self, func):
		return PairRDD([(x[0],func(x[1])) for x in self.data])

	def join(self, rdd):
		d1 = dict(self.data)
		d2 = dict(rdd.data)
		return PairRDD([(k, (d1[k], d2[k])) for k in d1])


def parallelize(iterator):
	x = iterator[0]
	if isinstance(x, tuple) and len(x) == 2:
		return PairRDD(iterator)
	else:
		return RDD(iterator)
