import os
from subprocess import call

DATASET_FILE = 'sets.txt'
PROGRAM = 'pypy ../code/builder.py %s %s'
CONF_LINES = 3

if __name__ == "__main__" :
	with open( DATASET_FILE , 'r' ) as f :
		lines = [ l[ :-1 ] for l in f.readlines() ]
		for i in xrange( 0 , len( lines ) , CONF_LINES ) :
			dataset , ommit , _ = lines[ i:(i+CONF_LINES) ]
			print "PROCESSING DATASET %s" % dataset
			inst = ( PROGRAM % ( dataset , ommit ) ).split()
			call( inst )
