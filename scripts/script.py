import os
from subprocess import call

DATASET_FILE = 'sets.txt'
PROGRAM = 'python ../code/builder.py %s %s %s %s'

if __name__ == "__main__" :
	with open( DATASET_FILE , 'r' ) as f :
		lines = [ l[ :-1 ] for l in f.readlines() ]
		for i in xrange( 0 , len( lines ) , 5 ) :
			train , test , ommit , out , _ = lines[ i:(i+5) ]
			inst = ( PROGRAM % ( train , test , ommit , out ) ).split()
			call( inst )
