from random import randint
import itertools

''' ======== FILES PARAMETERS ======== '''
DATA_DIR = '../data/'
RESULTS_DIR = '../results/'
TRAINING_FILE = DATA_DIR + 'training.csv'
TEST_FILE = DATA_DIR + 'test.csv'
MODELS = [ DATA_DIR + x for x in [ 'model1.txt' , 'model2.txt' , 'model3.txt' ] ]

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
GENERATED_DATA = 30000
GEN_TRAINING_FILE = DATA_DIR + 'gentraining_%s'
GEN_TEST_FILE = DATA_DIR + 'gentest_%s'
SIZE_TO_GET_RAND_VALUE = 100

''' ======== LEARNING PARAMETERS ======== '''
MAX_NUM_PARENTS = 3
NUM_ORDERING_SAMPLES = 100
NUM_GREEDY_RESTARTS = 5

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

def getsubconj( data ) :
	length = len( data )
	data = [ ( k , data[ k ] ) for k in data ]
	resp = []
	for i in range( 1 , length + 1 ) :
		resp.append( [ dict( list( x ) ) for x in itertools.combinations( data , i ) ] )
	return resp

if __name__ == "__main__" :
	a = { 'a': 1 , 'b' : 2 , 'c': 3 }
	x = getsubconj( a )
	for r in x : print r
