from random import randint
import subprocess as sub
import os
import itertools
import resource

''' ======== FILES PARAMETERS ======== '''
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'

''' ======== CONSTANTS ======== '''
FIELD_DELIMITER = ','

NUMERIC_FIELD = 'numeric'
LITERAL_FIELD = 'literal'

INT_MAX = int( 2e80 )

ESS = 1.0

EPSILON = 1e-7

''' ======== LEARNING PARAMETERS ======== '''
MAX_NUM_PARENTS = 3
NUM_RANDOM_RESTARTS = 3
NUM_GREEDY_ITERATIONS = 100
NUM_INITIAL_SOLUTIONS = 1000

def shuffle( arr ) :
	new_arr = list( arr )
	for i in xrange( len( new_arr ) - 1 , 0 , -1 ) :
		pos = randint( 0 , i )
		new_arr[ i ] , new_arr[ pos ] = new_arr[ pos ] , new_arr[ i ]
	return new_arr

def topological( graph , nodes ) :
	visited = {}
	order_fields = sorted( nodes , key = lambda node : len( graph[ node ][ 'parents' ] ) )
	indegree = dict( [ ( node , len( graph[ node ][ 'parents' ] ) ) for node in order_fields ] )
	topo_order = [ field for field in order_fields if indegree[ field ] == 0 ]
	topo_order = shuffle( topo_order )
	for node in topo_order :
		if node in visited : continue
		dfs( graph , node , visited , indegree , topo_order )
	return topo_order

def dfs( graph , node , visited , indegree , topo_order ) :
	visited[ node ] = True
	if node not in topo_order : topo_order.append( node )
	graph[ node ][ 'childs' ] = shuffle( graph[ node ][ 'childs' ] )
	for child in graph[ node ][ 'childs' ] :
		indegree[ child ] -= 1
		if indegree[ child ] == 0 :
			dfs( graph , child , visited , indegree , topo_order )

def compare( fa , fb ) :
	return -1 if fa + EPSILON < fb else 1 if fa - EPSILON > fb else 0

def getsubconj( data , keys ) :
	resp = dict( [ ( k , data[ k ] ) for k in keys ] )
	return resp
