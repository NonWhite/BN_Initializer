from random import randint
import subprocess as sub
import os
import itertools

''' ======== FILES PARAMETERS ======== '''
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'
TRAINING_FILE = DATA_DIR + 'training.csv'
TEST_FILE = DATA_DIR + 'test.csv'
MODELS = [ DATA_DIR + f for f in os.listdir( DATA_DIR ) if f.endswith( '.mdl' ) ]

''' ======== CONSTANTS ======== '''
FIELD_DELIMITER = ','

NUMERIC_FIELD = 'numeric'
LITERAL_FIELD = 'literal'

INT_MAX = int( 2e30 )

ESS = 1.0

EPSILON = 1e-7

''' ======== GENERATION PARAMETERS ======== '''
TRAINING_DATA_PERCENTAGE = 0.65
TEST_DATA_PERCENTAGE = 1 - TRAINING_DATA_PERCENTAGE
GENERATED_DATA = 5000
GEN_TRAINING_FILE = DATA_DIR + 'gentraining_%s'
GEN_TEST_FILE = DATA_DIR + 'gentest_%s'
SIZE_TO_GET_RAND_VALUE = 100

''' ======== LEARNING PARAMETERS ======== '''
MAX_NUM_PARENTS = 3
NUM_ORDERING_SAMPLES = 100
NUM_GREEDY_RESTARTS = 5
NUM_GREEDY_ITERATIONS = 5

''' ======== QUERY CSV COMMAND ======== '''
QUERY_CSV_COMMAND = './querycsv.py -i %s "select %s, count(*) from %s group by %s;"'

def shuffle( arr ) :
	new_arr = list( arr )
	for i in range( len( new_arr ) - 1 , 0 , -1 ) :
		pos = randint( 0 , i )
		new_arr[ i ] , new_arr[ pos ] = new_arr[ pos ] , new_arr[ i ]
	return new_arr

def topological( graph , nodes ) :
	visited = {}
	order_fields = sorted( nodes , key = lambda node : len( graph[ node ][ 'parents' ] ) )
	indegree = dict( [ ( node , len( graph[ node ][ 'parents' ] ) ) for node in order_fields ] )
	topo_order = [ field for field in order_fields if indegree[ field ] == 0 ]
	for node in topo_order :
		if node in visited : continue
		dfs( graph , node , visited , indegree , topo_order )
	return topo_order

def dfs( graph , node , visited , indegree , topo_order ) :
	visited[ node ] = True
	if node not in topo_order : topo_order.append( node )
	for child in graph[ node ][ 'childs' ] :
		indegree[ child ] -= 1
		if indegree[ child ] == 0 :
			dfs( graph , child , visited , indegree , topo_order )

def compare( fa , fb ) :
	return -1 if fa + EPSILON < fb else 1 if fa - EPSILON > fb else 0

def getsubconj( data , keys ) :
	resp = dict( [ ( k , data[ k ] ) for k in keys ] )
	return resp

def do_query_csv( csvfile , attributes ) :
	modelname = os.path.splitext( os.path.basename( csvfile ) )[ 0 ]
	attr_str = ','.join( attributes )
	print "Querying %s for attr: %s" % ( csvfile , attr_str )
	query = QUERY_CSV_COMMAND % ( csvfile , attr_str , modelname , attr_str )
	output = sub.check_output( query , stderr = sub.STDOUT , shell = True )
	return parse_query_response( output )

def parse_query_response( output ) :
	out = [ map( str.strip , line.split( '|' )  ) for line in output.split( '\n' ) ]
	fields = out[ 0 ][ :-1 ]
	num_fields = len( fields )
	out = out[ 2:-1 ]
	resp = []
	for row in out :
		data = dict( [ ( fields[ i ] , row[ i ] ) for i in range( num_fields ) ] )
		new_row = [ data , int( row[ num_fields ] ) ]
		resp.append( new_row )
	return resp

# TODO
def cpu_time() :
	return 0.0
