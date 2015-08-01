import os
import numpy as np
from pylab import *
from os.path import *
from copy import deepcopy as copy

color = [ 'b' , 'r' , 'g' ]
types = [ 'random' , 'unweighted' , 'weighted' ]
datasets = [ 'alarm' , 'census' , 'epigenetics' , 'image' , 'letter' , 'mushroom' , 'sensors' , 'steelPlates' ]
SOL_DELIMITER = ' =='
IMAGES_DIR = '../doc/images/'

def read_content( fpath , name ) :
	data = { 'name' : name , 'solutions' : [] , 'init_time' : 0 }
	sol = { 'score' : [] , 'iterations' : 0 , 'total_time' : 0 }
	print fpath
	with open( fpath , 'r' ) as f :
		lines = [ l[ :-1 ] for l in f.readlines() ]
		idx = 1
		if lines[ 0 ].startswith( 'TIME(init' ) :
			init_time = float( lines[ 0 ].split( ' = ' )[ -1 ] )
			data[ 'init_time' ] = init_time
		while idx < len( lines ) :
			if lines[ idx ].startswith( SOL_DELIMITER ) :
				solution = copy( sol )
				while True :
					line = lines[ idx ]
					if line.startswith( 'SCORE' ) :
						score = float( line.split( ' = ' )[ -1 ] )
						solution[ 'score' ].append( score )
					elif line.startswith( 'NUM IT' ) :
						iterations = int( line.split( ' = ' )[ -1 ] )
						solution[ 'iterations' ] = iterations
					elif line.startswith( 'TIME(total' ) :
						total_time = float( line.split( ' = ' )[ -1 ] )
						solution[ 'total_time' ] = total_time
					idx += 1
					if idx >= len( lines ) or \
						lines[ idx ].startswith( SOL_DELIMITER ) or \
						lines[ idx ].startswith( 'BEST' ) :
						break
				data[ 'solutions' ].append( solution )
			if lines[ idx ].startswith( 'BEST' ) : break
	max_val = max( s[ 'score' ][ -1 ] for s in data[ 'solutions' ] if len( s[ 'score' ] ) > 0 )
	for s in data[ 'solutions' ] :
		if len( s[ 'score' ] ) == 0 :
			s[ 'score' ].append( 0.0 )
	num_sols = sum( [ 1 for s in data[ 'solutions' ] if s[ 'score' ][ -1 ] == max_val ] )
	total_sols = len( data[ 'solutions' ] )
	max_iterations = max( [ s[ 'iterations' ] for s in data[ 'solutions' ] ] )
	avg_iterations = float( sum( [ s[ 'iterations' ] for s in data[ 'solutions' ] ] ) ) / total_sols
	max_total_time = max( [ s[ 'total_time' ] for s in data[ 'solutions' ] ] )
	avg_total_time = sum( [ s[ 'total_time' ] for s in data[ 'solutions' ] ] ) / total_sols
	print name.upper()
	#print "INITIALIZATION TIME = %s" % data[ 'init_time' ]
	#print "MAX TOTAL TIME = %s" % max_total_time
	#print "AVG TOTAL TIME = %s" % avg_total_time
	print "MAX SCORE = %s" % max_val
	print "TOTAL NUM SOLUTIONS = %s" % total_sols
	print "NUM OF SOLUTIONS WITH MAX SCORE = %s" % num_sols
	print "PERCENTAGE WITH MAX SCORE = %s" % ( float( num_sols ) / total_sols )
	print "MAX NUM ITERATIONS = %s" % max_iterations
	print "AVG NUM ITERATIONS = %s" % avg_iterations
	best_sols = [ s for s in data[ 'solutions' ] if s[ 'score' ][ -1 ] == max_val ]
	sorted( best_sols , key = lambda s : s[ 'iterations' ] , reverse = True )
	data[ 'score' ] = best_sols[ 0 ][ 'score' ]
	data[ 'iterations' ] = best_sols[ 0 ][ 'iterations' ]
	data.pop( 'solutions' )
	return data

def addCurve( x , y , col , lbl ) :
	style = '-'
	plot( x , y , color = col , linestyle = style , label = lbl )

def addPoint( x , y , col ) :
	plot( x , y , col+'o' )

def makePlot( directory , dataname ) :
	networkdata = []
	for t in types :
		f = "%s%s_%s.txt" % ( directory , dataname , t )
		networkdata.append( read_content( f , t ) )
	max_iterations = max( [ d[ 'iterations' ] for d in networkdata ] )
	for d in networkdata :
		missing = max_iterations - len( d[ 'score' ] )
		best = d[ 'score' ][ -1 ]
		for r in xrange( missing ) :
			d[ 'score' ].append( best )
	for i in range( len( networkdata ) ) :
		data = networkdata[ i ]
		y = data[ 'score' ]
		x = range( 1 , len( y ) + 1 )
		col = color[ i ]
		addCurve( x , y , col , data[ 'name' ] )
		#addPoint( data[ 'iterations' ] , data[ 'score' ][ -1 ] , col )
	legend( loc = 'lower right' )
	xlabel( 'Iteration' )
	ylabel( 'BIC Score' )
	savefig( "%s%s" % ( IMAGES_DIR , dataname ) )
	#show()
	clf()
	print ' ====================================================== '

if __name__ == "__main__":
	directory = '../results/'
	for d in datasets :
		makePlot( directory , d )
