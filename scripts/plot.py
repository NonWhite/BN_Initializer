from pylab import *
import os
from os.path import *
import numpy as np

color = [ 'blue' , 'purple' , 'green' , 'red' ]

def read_content( fpath ) :
	name = splitext( fpath.split( '/' )[ -1 ] )[ 0 ]
	name = name.replace( 'gentraining_' , '' )
	data = { 'name' : name , 'allscore' : [] , 'bestscore' : [] }
	with open( fpath , 'r' ) as f :
		for line in f :
			if not line.startswith( 'SCORE' ) : continue
			score = float( line[ :-1 ].split( ' = ' )[ -1 ] ) / 1000.0
			best = ( data[ 'bestscore' ][ -1 ] if data[ 'bestscore' ] else score )
			data[ 'allscore' ].append( score )
			data[ 'bestscore' ].append( max( score , best ) )
	return data

def addCurve( x , y , col , lbl ) :
	style = '-'
	plot( x , y , color = col , linestyle = style , label = lbl )

def makePlot( directory ) :
	files = [ directory + f for f in os.listdir( directory ) if f.endswith( '.txt' ) ]
	networkdata = []
	for f in files : networkdata.append( read_content( f ) ) 
	for i in range( len( networkdata ) ) :
		data = networkdata[ i ]
		y = data[ 'bestscore' ][ :100 ]
		x = range( 1 , len( y ) + 1 )
		col = color[ i ]
		addCurve( x , y , col , data[ 'name' ] )
	legend( loc = 'lower right' )
	xlabel( 'Iteration' )
	ylabel( 'BIC Score' )
	savefig( directory + 'bic_score' )
	clf()

if __name__ == "__main__":
	directory = '../results/'
	makePlot( directory )
