import os
import statistics
import numpy as np
from pylab import *
from os.path import *
from copy import deepcopy as copy

color = [ 'b' , 'r' , 'g' ]
#types = [ 'random' , 'unweighted' , 'weighted' ]
#datasets = [ 'alarm' , 'census' , 'epigenetics' , 'image' , 'letter' , 'lungCancer' , 'mushroom' , 'sensors' , 'spectf' , 'steelPlates' ]
datasets = [ 'census' , 'letter' , 'image' , 'mushroom' ]
types = [ 'random' , 'unweighted' , 'weighted' ]
SOL_DELIMITER = ' =='
IMAGES_DIR = '../doc/images/'

def read_content( fpath , name ) :
	data = { 'name' : name , 'solutions' : [] }
	sol = { 'score' : [] , 'iterations' : 0 }
	print fpath
	with open( fpath , 'r' ) as f :
		lines = [ l[ :-1 ] for l in f.readlines() ]
		idx = 0
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
					idx += 1
					if idx >= len( lines ) or \
						lines[ idx ].startswith( SOL_DELIMITER ) or \
						lines[ idx ].startswith( 'BEST' ) :
						break
				data[ 'solutions' ].append( solution )
			if idx >= len( lines ) or lines[ idx ].startswith( 'BEST' ) : break
	best_sc = [ s[ 'score' ][ -1 ] for s in data[ 'solutions' ] ]
	max_sc = max( best_sc )
	stdev_sc = statistics.stdev( best_sc )
	num_sols = sum( [ 1 for s in data[ 'solutions' ] if s[ 'score' ][ -1 ] == max_sc ] )
	total_sols = len( data[ 'solutions' ] )
	perc_sols = float( num_sols ) / total_sols * 100.0
	all_it = [ s[ 'iterations' ] for s in data[ 'solutions' ] ]
	max_iterations = max( all_it )
	avg_iterations = statistics.mean( all_it )
	stdev_iterations = statistics.stdev( all_it )
	print name.upper()
	print "MAX SCORE = %s" % max_sc
	print "STDEV MAX SCORE = %s" % stdev_sc
	print "TOTAL NUM SOLUTIONS = %s" % total_sols
	print "NUM OF SOLUTIONS WITH MAX SCORE = %s" % num_sols
	print "PERCENTAGE WITH MAX SCORE = %s" % perc_sols
	print "MAX NUM ITERATIONS = %s" % max_iterations
	print "AVG NUM ITERATIONS = %s" % avg_iterations
	print "STDEV NUM ITERATIONS = %s" % stdev_iterations
	print "%.3f +/- %.2f & %.2f & %.2f +/- %.2f" % ( max_sc , stdev_sc , perc_sols , avg_iterations , stdev_iterations )
	max_length = max( [ len( s[ 'score' ] ) for s in data[ 'solutions' ] ] )
	avg_scores = []
	for i in xrange( max_length ) :
		sc = 0.0
		q = 0
		for sol in data[ 'solutions' ] :
			if len( sol[ 'score' ] ) <= i : continue
			q += 1
			sc += sol[ 'score' ][ i ]
		avg_scores.append( sc / q )
	data[ 'score' ] = avg_scores
	data[ 'iterations' ] = len( avg_scores )
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
	for i in range( len( networkdata ) ) :
		data = networkdata[ i ]
		y = data[ 'score' ]
		x = range( 1 , len( y ) + 1 )
		col = color[ i ]
		addCurve( x , y , col , data[ 'name' ] )
	legend( loc = 'lower right' )
	xlabel( 'Iteration' )
	ylabel( 'BIC Score' )
	#savefig( "%s%s" % ( IMAGES_DIR , dataname ) )
	show()
	clf()
	print ' ====================================================== '

if __name__ == "__main__":
	directory = '../results/'
	for d in datasets :
		makePlot( directory , d )
