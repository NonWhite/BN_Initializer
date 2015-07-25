import os
from subprocess import call

DATASET_FILE = 'sets.txt'
PROGRAM = 'pypy ../code/builder.py %s %s %s'
CONF_LINES = 4

if __name__ == "__main__" :
	with open( DATASET_FILE , 'r' ) as f :
		lines = [ l[ :-1 ] for l in f.readlines() ]
		for i in xrange( 0 , len( lines ) , CONF_LINES ) :
			dataset , ommit , out , _ = lines[ i:(i+CONF_LINES) ]
			inst = ( PROGRAM % ( dataset , ommit , out ) ).split()
			call( inst )
